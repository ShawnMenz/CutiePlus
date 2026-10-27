"""
Messages over the cooperation module's Unix domain sockets: a window and the service, the service
and the GPU worker, the service and the root helper.

A message is an 8-byte big-endian length, a JSON header of that length, then the raw buffers the
header refers to, each with its own 8-byte length. numpy arrays and bytes travel as buffers; the
JSON holds {"__nd__": i, "dtype": ..., "shape": [...]} or {"__bytes__": i} in their place. Nothing
is unpickled, so a message cannot run code in the receiver. Only plain numeric dtypes are decoded.

A request is {"op": name, "args": {...}}; its reply {"ok": true, "result": ...} or
{"ok": false, "error": {"code": ..., "message": ..., ...}}.
"""
import json
import socket
import struct
from typing import Any, List

try:
    import numpy as np
except ImportError:  # the root helper runs on the standard library only; it sends no arrays
    np = None

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1 << 30
MAX_HEADER_BYTES = 16 << 20
MAX_BUFFERS = 4096
_LENGTH = struct.Struct('>Q')
# numpy arrays of these kinds only: no object arrays, whatever a peer sends
_ARRAY_KINDS = 'biuf'


def ok(result=None) -> dict:
    return {'ok': True, 'result': result}


def error(code: str, message: str, **details) -> dict:
    return {'ok': False, 'error': dict(details, code=code, message=message)}


class ProtocolError(Exception):
    pass


def _encode(value: Any, buffers: List[bytes]) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        buffers.append(array.tobytes())
        return {'__nd__': len(buffers) - 1, 'dtype': array.dtype.str, 'shape': list(array.shape)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        buffers.append(bytes(value))
        return {'__bytes__': len(buffers) - 1}
    if isinstance(value, dict):
        return {str(key): _encode(item, buffers) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, buffers) for item in value]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value: Any, buffers: List[bytes]) -> Any:
    if isinstance(value, dict):
        if '__nd__' in value:
            if np is None:
                raise ProtocolError('arrays are not accepted here')
            dtype = np.dtype(value['dtype'])
            if dtype.kind not in _ARRAY_KINDS:
                raise ProtocolError(f'arrays of {dtype} are not accepted')
            data = buffers[value['__nd__']]
            return np.frombuffer(data, dtype=dtype).reshape(value['shape']).copy()
        if '__bytes__' in value:
            return buffers[value['__bytes__']]
        return {key: _decode(item, buffers) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, buffers) for item in value]
    return value


def encode_message(message: Any) -> bytes:
    buffers: List[bytes] = []
    header = json.dumps(_encode(message, buffers), ensure_ascii=False).encode('utf-8')
    parts = [_LENGTH.pack(len(header)), header, _LENGTH.pack(len(buffers))]
    for buffer in buffers:
        parts.append(_LENGTH.pack(len(buffer)))
        parts.append(buffer)
    return b''.join(parts)


def send_message(sock: socket.socket, message: Any) -> None:
    sock.sendall(encode_message(message))


class _BytesReader:
    """recv_into over bytes, so a stored message decodes as a received one."""

    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.offset = 0

    def recv_into(self, view, size: int) -> int:
        count = min(size, len(self.data) - self.offset)
        view[:count] = self.data[self.offset:self.offset + count]
        self.offset += count
        return count


def decode_message(data: bytes) -> Any:
    return receive_message(_BytesReader(data))


def _receive_exactly(sock: socket.socket, size: int) -> bytes:
    if size > MAX_MESSAGE_BYTES:
        raise ProtocolError(f'message part of {size} bytes is too large')
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:], size - received)
        if count == 0:
            raise ConnectionError('the connection closed')
        received += count
    return bytes(data)


def receive_message(sock: socket.socket) -> Any:
    header_length = _LENGTH.unpack(_receive_exactly(sock, _LENGTH.size))[0]
    if header_length > MAX_HEADER_BYTES:
        raise ProtocolError(f'a header of {header_length} bytes is too large')
    try:
        header = json.loads(_receive_exactly(sock, header_length).decode('utf-8'))
    except ValueError as exc:  # also UnicodeDecodeError
        raise ProtocolError(f'the header is not JSON: {exc}') from None
    count = _LENGTH.unpack(_receive_exactly(sock, _LENGTH.size))[0]
    if count > MAX_BUFFERS:
        raise ProtocolError(f'{count} buffers in one message')
    buffers = []
    total = 0
    for _ in range(count):
        size = _LENGTH.unpack(_receive_exactly(sock, _LENGTH.size))[0]
        total += size
        if total > MAX_MESSAGE_BYTES:
            raise ProtocolError('the message is too large')
        buffers.append(_receive_exactly(sock, size))
    try:
        return _decode(header, buffers)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f'a malformed message: {exc}') from None


def request(sock: socket.socket, op: str, **args) -> Any:
    """Sends one request and returns its reply."""
    send_message(sock, {'op': op, 'args': args})
    return receive_message(sock)
