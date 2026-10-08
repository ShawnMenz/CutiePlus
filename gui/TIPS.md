### Cutie labeling guide

**Workflow**

1. Open a sequence from the task browser.
2. Fix frames with clicks (or Reset), and mark good frames as anchors (C).
3. Propagate (F forward, B backward). Anchors are never overwritten; propagation stops at "target absent" frames.
4. Save binary PNGs (Ctrl+S): the sequence goes "under review". A reviewer accepts it with Mark done in the task browser.

**Mouse**

- Left click: add to the target (green dot).
- Right click: remove from the target (red dot).
- Ctrl + wheel: zoom in / out at the cursor (back to fit at the smallest zoom).
- Wheel / Alt + wheel: scroll up-down / left-right when zoomed in.

The dots belong to the current frame's clicks. They clear when you change frame, mark an anchor, or reset.

**Keyboard**

| Key | Action |
|---|---|
| A / D | previous / next frame (the arrow keys do not move frames) |
| Shift + A / D | 10 frames back / forward |
| Alt + A / D | first / last frame |
| Ctrl + A / D | previous / next anchor frame (also [ / ]) |
| { / } (Shift + [ / ]) | previous / next interval start; inside an interval { goes to its own start first; at the ends they wrap around |
| R | play / stop the video: it plays once and stops on the last frame |
| Shift + R | preview the intervals: each press plays the next interval only, from its start to its end |
| F / B | propagate forward / backward (press again to pause) |
| Space | cycle davis → mask → rgba → light visualization (other modes remain in the menu) |
| C | mark this frame as an anchor |
| Ctrl+S | Save binary PNGs (submits the sequence for review) |
| Ctrl+E | Save overlay video |
| Delete | remove this frame's anchor (back to its prefill) |
| T | show RGB / thermal |
| Ctrl+Z | undo the last click, otherwise the last frame edit |
| Ctrl+Shift+Z or Ctrl+Y | redo |
| Q | close this window |

Undo covers clicks, Reset frame, Import mask, C and Delete. Propagation cannot be undone; propagate again instead.

**Top row**

- The target: the category, and the description in a text box. It starts as the curated text; edit it to tune the wording. Press Enter or click elsewhere to save (it cannot be empty). Changing it needs a new review. Save binary PNGs also writes it as `description.txt` (the curated CSV itself is not updated).
- Frame label, e.g. `000020.jpg (RGB) [anchor: ls]`: the modality shown (RGB, with a red R, green G and blue B; or thermal, in orange-red) and, in color, where the mask comes from:
  - anchor (purple): ls / ls_refined / cutie. A human decision, never overwritten.
  - empty anchor (magenta): target absent. Propagation stops here.
  - propagated (teal): written by Cutie.
  - prefill (olive-gold): sam3 / bb2seg. Not reviewed yet.
  - no mask (gray): nothing yet.
- Intervals (the row under the description): the dataset clip, start and end in frame numbers. It is preset from the curated CSV (long sequences: all frames). Press Enter or click elsewhere to save. Changing it needs a new review. The saved intervals are shown as bands on the frame slider; { and } jump to the previous / next interval start. Type `m` in a start box for the first frame, or in an end box for the last frame: once you confirm, the `m` becomes the frame number (e.g. `1` and `139` for a 139-frame sequence).
- Several intervals (optional): when a sequence has more than one good stretch, press **+ Add interval** and type the new start and end; each interval becomes a clip of its own. The new interval is saved once both boxes are filled. **×** cancels an interval that is not complete, or removes a saved one. Intervals may touch (to split a clip in two) but not overlap: an overlapping one turns red and is not saved until you change it. To drop the first interval, type the second one's numbers into it and remove the second.
- Keyboard: the frame slider has it when the window opens, so A / D move frames (the two boxes beside the slider name them; hover over one for the Shift, Alt and Ctrl combinations). The slider also takes Home / End (first / last frame) and PageUp / PageDown (10 frames). While the description or a number box has the cursor, the keys type into it; Enter or a click on the image gives them back to the slider.

**Bottom controls**

Three groups, left to right; in a narrow window they wrap onto more rows.

*Frame*

- Counter (e.g. 20 / 90), Play video (R), Reset frame (whole frame empty = "target absent" anchor), Reset object.
- Current object ID: tasks have one object.
- Play video (R) plays from the current frame to the last one and stops there (it does not loop). Press R or the button again to stop early; at the last frame it plays again from the first. Shift + R (or Shift + click on the button) previews only the intervals: the first press plays interval 1 and stops at its end, the next press interval 2, and so on, then the first again. After { or }, Shift + R plays the interval jumped to.

*Display and files*

- Visualization mode: how the mask is drawn on screen.
- Save binary PNGs (Ctrl+S): writes binary PNGs of the saved masks, `interval.txt` and `description.txt` into `binary_masks/`, and submits the sequence for review. These files are not the dataset; Export task masks writes that.
- Save overlay video (Ctrl+E): a video from saved visualizations (not the dataset). Output FPS and bitrate apply to it.
- Save visualization and Save soft mask during propagation: write a file per frame into the task folder. Keep them off unless needed.

*Labeling*

- Mark as anchor (C).
- Export task masks: writes the dataset masks now. The status does not change; Mark done in the browser exports anyway.
- Propagate forward (F) / backward (B): while running, the button reads "Pause propagation" with the same key.

**Right panel**

- Tips (F1): opens this guide in a window of its own, beside the labeling window. Esc closes it.
- The log at the bottom of the panel takes the height that is left, so it grows and shrinks with the window.
- It scrolls when the window is short. Drag its left edge to resize it, or to the window's right edge to hide it; drag it back from there.
- Memory gauges: permanent (your anchors), working, long-term, and GPU.
- Reset all memory / Reset non-permanent memory: clears the model's memory; masks on disk stay. Rarely needed, since each propagation rebuilds memory from the anchors.
- Reset all frame masks (red text): after confirmation, deletes this sequence's local, binary, and soft mask PNGs. In task mode it also clears anchors and prefills, deletes both Cutie export trees, and returns the task to "in progress". This cannot be undone; the button is disabled when the curated CSV marks the sequence `done`.
- Working and long-term memory settings: model tuning. The defaults are fine.
- Import mask: loads a mask image into this frame and makes it an anchor. Import layer: a background image for the "layer" view.
- Small targets are processed at 960 px instead of 480 px; the console says which on start-up.

**Status in the task browser**

not started, then in progress (any change), then under review (Save binary PNGs), then done (Mark done in the browser, which exports first). Any change reopens a sequence. Hover over a status to see who last edited it. Tick Hide done (next to the filter box) to hide the done sequences in every tab; the counts at the bottom still include them. In the Target column the category is in bold.

**Several labelers at once**

- Open by: who has the sequence open, and on which display ("exporting" while a browser exports it). It updates every 10 seconds.
- A sequence open elsewhere cannot be opened, exported, or marked done until it is closed there. A window that crashes releases its sequence at once.
- Your name (in the window title) is recorded on your anchors and review steps: `--labeler`, else the annotator assigned to the port you connect through (Manage annotators), else `CUTIE_LABELER`, else the account name.
- Annotators: everyone who changed the sequence (any change; Save binary PNGs and Mark done do not count). The bottom line counts the sequences each annotator worked on in this tab, whatever their status now; a sequence with two annotators counts for both. Type a name in the filter to list that person's sequences.
- Manage annotators (button in the task browser): who connects to which port. Each row is `Port: [number] --> [name]`; port 59NN is virtual screen :NN. Add, edit and remove rows, look one up with the box on top, then press **Save**: nothing is written before that. The list is kept in `annotators.json` in the task root, and the next sequence you open on any screen records the new names, without a restart (a window that is already open keeps its name). A screen with no entry records the account name.

Outside task mode (a plain Cutie workspace), C commits the frame to permanent memory, and the save buttons read "Export binary masks" and "Export as video" (same keys, Ctrl+S and Ctrl+E).
