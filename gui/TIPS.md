### Cutie labeling guide

**Workflow**

1. Open a sequence from the task browser.
2. Fix frames with clicks (or Reset), and mark good frames as anchors (C).
3. Propagate (F forward, B backward). Anchors are never overwritten; propagation stops at "target absent" frames. Started inside an interval, it stays there: F stops at the interval's last frame, B at its first. It runs on the modality shown: press T first to propagate on the thermal (or RGB) frames, with every anchor you have labeled.
4. Save binary PNGs (Ctrl+S): the sequence goes "under review". A reviewer accepts it with Mark done in the task browser.

**Mouse**

- Left click (or , with the pointer on the image): add to the target (green dot).
- Right click (or . with the pointer on the image): remove from the target (red dot).
- Ctrl + wheel (or Ctrl + + / Ctrl + -): zoom in / out at the cursor (back to fit at the smallest zoom); Ctrl + 0 shows the whole image again.
- Wheel / Alt + wheel (or the arrow keys): scroll up-down / left-right when zoomed in.

Without a mouse (a touchpad): move the pointer with the touchpad and press , and . instead of the buttons; the keys work alongside the mouse.

The dots belong to the current frame's clicks. They clear when you change frame, mark an anchor, or reset.

**Keyboard**

| Key | Action |
|---|---|
| A / D | previous / next frame (the arrow keys do not move frames: they scroll a zoomed image) |
| Shift + A / D | 10 frames back / forward |
| Alt + A / D | first / last frame |
| Ctrl + A / D | previous / next anchor frame (also [ / ]) |
| { / } (Shift + [ / ]) | previous / next interval start; inside an interval { goes to its own start first; at the ends they wrap around |
| R | play / stop the video: it plays once and stops on the last frame |
| Shift + R | preview the intervals: each press plays the next interval only, from its start to its end |
| F / B | propagate forward / backward (press again to pause); inside an interval, up to its last / first frame |
| Space | cycle davis → mask → rgba → light visualization (other modes remain in the menu) |
| C | mark this frame as an anchor |
| Ctrl+S | Save binary PNGs (submits the sequence for review) |
| Ctrl+E | Save overlay video |
| Delete | remove this frame's anchor (back to its prefill) |
| T | show RGB / thermal; F / B then propagate on the frames shown |
| Ctrl+Z | undo the last click, otherwise the last frame edit |
| Ctrl+Shift+Z or Ctrl+Y | redo |
| , / . | left / right click at the pointer: add to / remove from the target |
| Ctrl + + / Ctrl + - / Ctrl + 0 | zoom in / out / show the whole image |
| Arrow keys | scroll the image when zoomed in |
| Q | close this window |

Undo covers clicks, Reset frame, Import mask, C and Delete. Propagation cannot be undone; propagate again instead.

**Top row**

- The target: the category (the green pill), and the description in a text box. It starts as the curated text; edit it to tune the wording. Press Enter or click elsewhere to save (it cannot be empty). Changing it needs a new review. It describes the sequence and its first interval (the one on its right). Save binary PNGs also writes it as `description.txt` (the curated CSV itself is not updated).
- Frame label, e.g. `000020.jpg (RGB) [anchor: ls]`: the modality shown (RGB, with a red R, green G and blue B; or thermal, in orange-red) and, in color, where the mask comes from:
  - anchor (purple): ls / ls_refined / cutie. A human decision, never overwritten.
  - empty anchor (magenta): target absent. Propagation stops here.
  - propagated (teal): written by Cutie.
  - prefill (olive-gold): sam3 / bb2seg. Not reviewed yet.
  - no mask (gray): nothing yet.
- Intervals (right of the description): the dataset clip, start and end in frame numbers. It is preset from the curated CSV (long sequences: all frames). Press Enter or click elsewhere to save. Changing it needs a new review. The saved intervals are shown as bands on the frame slider; { and } jump to the previous / next interval start. Type `m` in a start box for the first frame, or in an end box for the last frame: once you confirm, the `m` becomes the frame number (e.g. `1` and `139` for a 139-frame sequence).
- Several intervals (optional): when a sequence has more than one good stretch, press **+ Add interval** and type the new start and end; each interval becomes a clip of its own. The new interval gets a row of its own, with its own description on the left. That description starts as the text in the top box: edit it when the clip needs other words, then press Enter or click elsewhere (it cannot be empty; a change needs a new review). Later edits of the top box do not change it. The new interval is saved, with its description, once both boxes are filled. **- Remove interval** (right of its interval) cancels an interval that is not complete, or removes a saved one with its description. Intervals may touch (to split a clip in two) but not overlap: an overlapping one turns red and is not saved until you change it. To drop the first interval, type the second one's numbers into it and remove the second. { / } and Shift+R show the interval's description in the messages. Save binary PNGs writes one description per interval in `description.txt` (`description_1=`, `description_2=`, … numbered as in `interval.txt`).
- Anchors on the frame slider: a small hollow black ring (white in the dark theme) at each anchor, a magenta ring at each empty anchor; they follow every change (a click, C, Delete, undo).
- Keyboard: the frame slider has it when the window opens, so A / D move frames (the two boxes beside the slider name them; hover over one for the Shift, Alt and Ctrl combinations). The slider also takes Home / End (first / last frame) and PageUp / PageDown (10 frames). While the description or a number box has the cursor, the keys type into it; Enter or a click on the image gives them back to the slider.
- Editing a description: the usual text keys work, and no window key acts while you type (no frame, anchor, zoom or mask change) except Ctrl+S (save, which also saves the text) and F1. Ctrl+C / Ctrl+X / Ctrl+V copy, cut and paste (also with your own computer, through the VNC viewer); Ctrl+Z undoes the typing, Ctrl+Shift+Z or Ctrl+Y redoes it; Ctrl+A selects all; Home / End (or ↑ / ↓) go to the start / end; Ctrl + ← / → move by word; Shift with any of them selects; Ctrl+Backspace / Ctrl+Delete delete a word; right-click opens the edit menu. From a Mac: the same keys with ⌃ Control, or ⌘ Command if the VNC viewer passes it on (⌘C / ⌘X / ⌘V, ⌘Z / ⌘⇧Z, ⌘A, ⌘← / ⌘→, ⌘⌫), ⌥ Option + ← / → move by word and ⌥⌫ deletes one. Avoid ⌥⇧ + arrows: the remote desktop moves the window to another workspace.

**Bottom controls**

Three cards under the frame slider, left to right; in a narrow window they wrap onto more rows. The main actions are the filled green buttons.

*Frame*

- Counter (e.g. 20 / 90), Play video (R) with its speed box (fps), Reset frame (whole frame empty = "target absent" anchor), Reset object.
- Current object ID: tasks have one object.
- Play video (R) plays from the current frame to the last one and stops there (it does not loop). Press R or the button again to stop early; at the last frame it plays again from the first. Shift + R (or Shift + click on the button) previews only the intervals: the first press plays interval 1 and stops at its end, the next press interval 2, and so on, then the first again. After { or }, Shift + R plays the interval jumped to. The box beside the button sets the speed of both, in frames per second (1-60, 30 to start); a change applies at once, also while playing, and is remembered for your screen.

*Propagate*

- Propagate backward (B) / forward (F): while running, the button reads "Pause propagation" with the same key. From a frame inside an interval, the run stays in that interval (use { / } to go to another one); from a frame outside every interval, it runs to the first / last frame of the sequence. The bar beside Mark as anchor shows the progress.
- Mark as anchor (C).

*Save*

- Save binary PNGs (Ctrl+S): writes binary PNGs of the saved masks, `interval.txt` and `description.txt` into `binary_masks/`, and submits the sequence for review. These files are not the dataset; Export task masks writes that.
- Export task masks: writes the dataset masks now. The status does not change; Mark done in the browser exports anyway.
- Need Attention (orange outline): click it when the sequence needs further attention, for example a problem you cannot solve or a question for someone else. While it is on, the button is filled orange, the task browser shows the status Attention in red, and Mark done refuses the sequence. Click again to clear it. Edits and Save binary PNGs leave it as it is; hover over the button to see who turned it on and when.
- Visualization mode: how the mask is drawn on screen.

**Sidebar**

- Messages: what the window did, newest at the bottom; it takes the height that is left. Tips (F1), at its top right, opens this guide in a window of its own, beside the labeling window. Esc closes it.
- The sidebar scrolls when the window is short. Drag its left edge to resize it, or to the window's right edge to hide it; drag it back from there.
- Advanced (click the title to open or close it; it stays as you left it on your screen): the controls you rarely need.
  - Memory gauges: permanent (your anchors), working, long-term, and GPU.
  - Reset all memory / Reset non-permanent memory: clears the model's memory; masks on disk stay. Rarely needed, since each propagation rebuilds memory from the anchors.
  - Working and long-term memory settings: model tuning. The defaults are fine.
  - Import mask: loads a mask image into this frame and makes it an anchor. Import layer: a background image for the "layer" view.
  - Overlay video: Save visualization and Save soft mask during propagation write a file per frame into the task folder (keep them off unless needed); Save overlay video (Ctrl+E) makes a video from the saved visualizations (not the dataset), with the Output FPS and bitrate.
  - Reset all frame masks (red outline): after confirmation, deletes this sequence's local, binary, and soft mask PNGs. In task mode it also clears anchors and prefills, deletes both Cutie export trees, and returns the task to "in progress". This cannot be undone; the button is disabled when the curated CSV marks the sequence `done`.
- Small targets are processed at 960 px instead of 480 px; the messages say which on start-up.

**Status in the task browser**

not started, then in progress (any change), then under review (Save binary PNGs), then done (Mark done in the browser, or the D key there, which exports first). Any change reopens a sequence. A sequence whose Need Attention is on shows Attention (red) instead, whatever its review status, until someone turns it off in the labeling window. Hover over a status to see who last edited it (and who asked for attention). Discard (beside Mark done) leaves the selected sequence out of the dataset after you confirm: its status is discarded (gray, italics), its done mark and Need Attention are removed, and it cannot be exported or marked done; its masks and anchors stay. On a discarded sequence the button reads Restore and brings it back (it then needs Save binary PNGs or Mark done again). If its row in the curated CSV already says done, it stays in the dataset until that row changes; the browser tells you. Tick Hide done / discarded (next to the filter box) to hide the done and discarded sequences in every tab; the counts in the Statistics card still include them. In the Target column the category is in bold.

Frames shows the sequence's frame count until it is marked done; then the frames of each of its intervals, in order, e.g. 120/85/40 (one number when it has one interval, the full count when it has none). Hover over it for the frame ranges. A click on its header does not sort; the other columns sort as before.

The Theme button (top right) cycles Auto, Light, Dark and Eye care, a light theme with a soft cream (#FAF9DE) in place of white.

**Several labelers at once**

- Open by: who has the sequence open, and on which display ("exporting" while a browser exports it). It updates every 10 seconds.
- A sequence open elsewhere cannot be opened, exported, or marked done until it is closed there. A window that crashes releases its sequence at once.
- Your name (in the window title) is recorded on your anchors and review steps: `--labeler`, else the annotator assigned to the port you connect through (Manage annotators), else `CUTIE_LABELER`, else the account name.
- Annotators: everyone who changed the sequence (any change; Save binary PNGs and Mark done do not count). The Statistics card (bottom right, beside the messages) counts the sequences each annotator worked on in this tab, whatever their status now; a sequence with two annotators counts for both. Type a name in the filter to list that person's sequences.
- Total Intervals (the last line of the Statistics card): the intervals of every sequence marked done, in all tabs (main and long), whatever tab is shown. Each interval is a clip of the dataset; a sequence with no interval of its own counts its curated one. A done sequence later flagged Need Attention still counts; a discarded one does not.
- Manage annotators (button in the task browser): who connects to which port. Each row is `Port: [number] --> [name]`; port 59NN is virtual screen :NN. Add, edit and remove rows, look one up with the box on top, then press **Save**: nothing is written before that. The list is kept in `annotators.json` in the task root, and the next sequence you open on any screen records the new names, without a restart (a window that is already open keeps its name). A screen with no entry records the account name.

Outside task mode (a plain Cutie workspace), C commits the frame to permanent memory, and the save buttons read "Export binary masks" and "Export as video" (same keys, Ctrl+S and Ctrl+E).
