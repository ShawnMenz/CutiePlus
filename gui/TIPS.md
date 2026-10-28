### Cutie labeling guide

**Workflow**

1. Open a sequence from the task browser.
2. Fix frames with clicks (or Reset), and mark good frames as anchors (C).
3. Propagate (F forward, B backward). Anchors are never overwritten; propagation stops at "target absent" frames. Started inside an interval, it stays there: F stops at the interval's last frame, B at its first. It runs on the modality shown: press T first to propagate on the thermal (or RGB) frames, with every anchor you have labeled.
4. Save (Ctrl+S): the sequence goes "under review". A reviewer accepts it with Mark done in the task browser. (Masks are saved as you label.)
   When you label and review the sequence at the same time, Ctrl+Q does steps 4 and Mark done in one go: it saves, marks the sequence done and closes the window (it asks first).
   Output video (Ctrl+E) also saves and marks the sequence done, once it has written the sequence's demo videos; the window stays open.
5. Export (task browser): every interval of every done sequence becomes a clip of the dataset in `workspace/cutie_export/`.

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
| Shift + R | preview the intervals: each press plays the next interval only, from its start to its end, and selects it |
| F / B | propagate forward / backward (press again to pause); inside an interval, up to its last / first frame |
| Space | cycle davis → mask → rgba → light visualization (other modes remain in the menu) |
| C | mark this frame as an anchor |
| Ctrl+S | Save (submits the sequence for review) |
| Delete | remove this frame's anchor (back to its pregenerated mask, if it has one) |
| T | show RGB / thermal; F / B then propagate on the frames shown |
| Ctrl+Z | undo the last click, otherwise the last frame edit |
| Ctrl+Shift+Z or Ctrl+Y | redo |
| , / . | left / right click at the pointer: add to / remove from the target |
| Ctrl + + / Ctrl + - / Ctrl + 0 | zoom in / out / show the whole image |
| Arrow keys | scroll the image when zoomed in |
| Q | close this window |
| Ctrl+Q | quit and mark done: save, mark the sequence done and close the window, after a warning (for labeling and review at the same time) |
| Ctrl+E | output video: save, write the demo videos, mark the sequence done |

Undo covers clicks, Reset frame, Import mask, C and Delete. Propagation cannot be undone; propagate again instead.

**Top row**

- The target: the category (a pill in the category's color; a gray `null` when there is none), and the description in a text box.
- The mask of the target is drawn in its category's color (cyan when it has none); the click dots have a thin white ring so they stay visible on it.
- Click the category pill to choose another: a menu lists the subset's categories, the name on the left and its color on the right. Type the first letters in its search box to keep the categories that start with them (`p`: person; `ca`: cable car, car, cat). ↓ / ↑ move, Enter or a click picks, Esc closes without a change. The choice is saved with the sequence's work (its `metadata.json`; the input's `targets.json` is not changed) and needs a new review. It starts as the input's text (`targets.json`; empty without one); edit it to tune the wording. Press Enter or click elsewhere to save (it cannot be empty). Changing it needs a new review. It describes the sequence and its first interval (the one on its right). It is saved with the sequence's work (the input's `targets.json` is not changed).
- Frame label, e.g. `000020.jpg (RGB) [anchor]`: the modality shown (RGB, with a red R, green G and blue B; or thermal, in orange-red) and, in color, where the mask comes from:
  - anchor (purple): a human decision, never overwritten.
  - empty anchor (magenta): target absent. Propagation stops here.
  - propagated (teal): written by Cutie.
  - pregenerated (olive-gold): a mask that came with the input (its `annotation/` folder). Not reviewed yet; shown until the frame gets a mask of its own.
  - no mask (gray): nothing yet.
- Intervals (right of the description): the dataset clip, start and end in frame numbers. It starts as the whole sequence. Press Enter or click elsewhere to save. Changing it needs a new review. The saved intervals are shown as bands on the frame slider: the selected one in bold green, the others in light green; { and } jump to the previous / next interval start and select that interval. Type `m` in a start box for the first frame, or in an end box for the last frame: once you confirm, the `m` becomes the frame number (e.g. `1` and `139` for a 139-frame sequence). On the right of the boxes, every interval shows its frame count, its first and last frames included (15 - 650: 636 frames); it follows the numbers as you type.
- Several intervals (optional): when a sequence has more than one good stretch, press **+ Add interval** and type the new start and end; each interval becomes a clip of its own, with a category and a description of its own. The window shows one set at a time (category, description, interval): the triangle at the left (▶ 1/3: set 1 of 3) opens the list of every set, one row each with its category, description and frames (**Interval:** 50-177); click a row to show and edit that set (the list folds, and the frame goes to its interval's start). A new set is shown at once; its category and description start as the sequence's (set 1's): click its pill to choose another category, edit its description and press Enter or click elsewhere (it cannot be empty; a change needs a new review). Later edits of set 1 do not change them. The new interval is saved, with its category and description, once both boxes are filled. The mask is drawn in the shown set's category color. **- Remove interval** (right of its interval) cancels an interval that is not complete, or removes a saved one with its description. Intervals may touch (to split a clip in two) but not overlap: an overlapping one turns red and is not saved until you change it. To drop the first interval, type the second one's numbers into it and remove the second. { / } and Shift+R show the interval's description in the messages. Each interval's category and description go with its clip into the export's seqinfo.
- Anchors on the frame slider: a small hollow black ring (white in the dark theme) at each anchor, a magenta ring at each empty anchor; they follow every change (a click, C, Delete, undo).
- Keyboard: the frame slider has it when the window opens, so A / D move frames (the two boxes beside the slider name them; hover over one for the Shift, Alt and Ctrl combinations). The slider also takes Home / End (first / last frame) and PageUp / PageDown (10 frames). While the description or a number box has the cursor, the keys type into it; Enter or a click on the image gives them back to the slider.
- Editing a description: the usual text keys work, and no window key acts while you type (no frame, anchor, zoom or mask change) except Ctrl+S (save, which also saves the text) and F1. Ctrl+C / Ctrl+X / Ctrl+V copy, cut and paste; Ctrl+Z undoes the typing, Ctrl+Shift+Z or Ctrl+Y redoes it; Ctrl+A selects all; Home / End (or ↑ / ↓) go to the start / end; Ctrl + ← / → move by word; Shift with any of them selects; Ctrl+Backspace / Ctrl+Delete delete a word; right-click opens the edit menu. On a Mac keyboard: the same keys with ⌃ Control, or with ⌘ Command (⌘C / ⌘X / ⌘V, ⌘Z / ⌘⇧Z, ⌘A, ⌘← / ⌘→, ⌘⌫); ⌥ Option + ← / → move by word and ⌥⌫ deletes one.

**Bottom controls**

Three cards under the frame slider, left to right; in a narrow window they wrap onto more rows. The main actions are the filled green buttons.

*Frame*

- Counter (e.g. 20 / 90), Play video (R) with its speed box (fps) and Lock, Reset frame (the frame's mask emptied: a "target absent" anchor). A window labels one object, the target.
- Play video (R) plays from the current frame to the last one and stops there (it does not loop). Press R or the button again to stop early; at the last frame it plays again from the first. Shift + R (or Shift + click on the button) previews only the intervals: the first press plays interval 1 and stops at its end, the next press interval 2, and so on, then the first again. After { or }, Shift + R plays the interval jumped to. The box beside the button sets the speed of both, in frames per second (1-60); a change applies at once, also while playing. Every window starts at 24.
- Lock (right of the speed box) keeps the speed: set it, then press Lock (it turns green and reads Locked, and the box can no longer change). Every window you open from then on starts at that speed. Press it again to unlock: the speed goes back to 24, here and in the next windows.

*Propagate*

- Propagate backward (B) / forward (F): while running, the button reads "Pause propagation" with the same key. From a frame inside an interval, the run stays in that interval (use { / } to go to another one); from a frame outside every interval, it runs to the first / last frame of the sequence. The bar beside Mark as anchor shows the progress.
- Mark as anchor (C).

*Save*

- Save (Ctrl+S): masks are saved as you label, each edit at once; Save also saves the intervals and descriptions as typed, and submits the sequence for review. The sequence's work folder holds only `masks/` and `metadata.json`.
- Quit and mark done (Ctrl+Q, no button): for a sequence you label and review yourself. A warning first shows the intervals: once you confirm, the window saves (with the intervals and descriptions), marks the sequence done in your name and closes. There is no second check by a reviewer, so look at every interval, its description and its masks before you press it. The time you pressed it is kept with the done mark. Need Attention, and an interval that is not complete or overlaps, stop it before the warning.
- Need Attention (orange outline): click it when the sequence needs further attention, for example a problem you cannot solve or a question for someone else. While it is on, the button is filled orange, the task browser shows the status Attention in red, and Mark done refuses the sequence. Click again to clear it. Edits and Save leave it as it is; hover over the button to see who turned it on and when.
- Output video (Ctrl+E) and Output FPS: writes one demo video per interval to `workspace/cutie_demo/<tab>/<sequence>_<start>-<end>_<mode>.mp4`: the RGB frame on the left and the thermal one on the right, both with the mask drawn in the current Visualization mode (davis unless you change it) and the interval's category color, the frame number of the clip in a band above (e.g. 12/101), and "category: description" in a band below, the category in its color. It saves first, and marks the sequence done in your name once every video is written (the window stays open). While it writes, the button reads Cancel output. Need Attention, a discarded sequence, a running or queued propagation, or an interval that is not complete stop it, as they stop Ctrl+Q; a change while it writes keeps the videos but not the done mark. Output again replaces the sequence's videos of that mode. Output FPS is remembered for the next window.
- Visualization mode: how the mask is drawn on screen, and in the output videos.

**Sidebar**

- Messages: what the window did, newest at the bottom; it takes the height that is left. Tips (F1), at its top right, opens this guide in a window of its own, beside the labeling window. Esc closes it.
- The sidebar scrolls when the window is short. Drag its left edge to resize it, or to the window's right edge to hide it; drag it back from there.
- Advanced (click the title to open or close it; it stays as you left it): the controls you rarely need.
  - Memory gauges: permanent (your anchors), working, long-term, and GPU. The first three show the memory of your latest propagation, live during the run and as the run left it afterwards.
  - Reset all memory / Reset non-permanent memory: clears the model's memory; masks on disk stay. Rarely needed, since each propagation rebuilds memory from the anchors.
  - Working and long-term memory settings: model tuning. The defaults are fine.
  - Import mask: loads a mask image into this frame and makes it an anchor.
  - Reset all frame masks (red outline): after confirmation, deletes this sequence's masks. In task mode it also clears the anchors and returns the task to "in progress"; the pregenerated masks of the input show again. This cannot be undone.
- Small targets are processed at 960 px instead of 480 px; the messages say which on start-up.

**Status in the task browser**

not started, then in progress (any change), then under review (Save), then done (Mark done in the browser, or the D key there). Any change reopens a sequence. A sequence whose Need Attention is on shows Attention (red) instead, whatever its review status, until someone turns it off in the labeling window. Hover over a status to see who last edited it (and who asked for attention). Discard (beside Mark done) leaves the selected sequence out of the dataset after you confirm: its status is discarded (gray, italics), its done mark and Need Attention are removed, and it cannot be marked done; its masks and anchors stay. On a discarded sequence the button reads Restore and brings it back (it then needs Save or Mark done again). Tick Hide done / discarded (next to the filter box) to hide the done and discarded sequences in every tab; the counts in the Statistics card still include them. A list with nothing to show says why in its middle: no data imported yet, every sequence done or discarded (with Hide done / discarded ticked), or none matching the filter. In the Target column the category is in bold (`null` when the input names none).

**Tabs and Import**

Each tab is a subset of the input: a folder in `workspace/cutie_tasks/` holding `visual/`, `thermal/` and `categories.json` (needed; it may list names only, e.g. `["car", "person"]`: Save then picks a distinct color for each and writes it in), and `targets.json` (each sequence's category and description), `annotation/` (pregenerated masks) and `seqinfo/` (optional). Import (beside Refresh) opens a window that lists the subsets, each checked: a ✓ line counts its sequences, targets, masks and categories, a ✗ line says what is missing, a ⚠ line warns. When the input sits directly in `workspace/cutie_tasks`, the first row is `main`: Save moves it into `main/`. + Add a subset adds a row: a tab name (it follows the folder's name until you type one) and a folder, typed or picked with Browse…; Save moves that folder into `workspace/cutie_tasks/<tab>` (instant on the same disk, a copy and then a delete from another disk). Save works once every row has a ✓; the new tabs appear when it is done. A sequence's work is kept apart, in `workspace/cutie_output/<tab>/<sequence>/`, made when it is first opened: `masks/` (its saved masks) and `metadata.json` (its source, category, description, intervals, review progress, Need Attention, discard and anchors).

**Export**

Export (beside Import) writes the dataset into `workspace/cutie_export/`, one folder for all tabs: each interval of each done sequence (a discarded one never) is a clip, numbered from 1 in the browser's order (tabs, then sequences, then intervals), zero-padded to the number of clips (3000 clips: 0001 … 3000). A clip has `visual/<clip>/` and `thermal/<clip>/` (the interval's frames, copied and numbered from 000001), `annotation/<clip>/` (its masks, black and white: the saved mask of each frame, else its pregenerated mask, else empty) and `seqinfo/<clip>_seqinfo.ini`. The seqinfo names where the clip comes from: `originalSource` is the tab and sequence (e.g. `main/0239`), shared by every clip cut from it, and `intervalStart` / `intervalEnd` are frames of that sequence; it also has the interval's category and description, its length and the image sizes. Export asks first, with the number of clips and the space the frames take; a sequence open in a labeling window stops it (close the window first). It replaces the previous export only once complete: Cancel, or a failure, leaves the previous one as it was.

Frames shows the sequence's frame count until it is marked done; then the frames of each of its intervals, in order, e.g. 120/85/40 (one number when it has one interval, the full count when it has none). Hover over it for the frame ranges. A click on its header does not sort; the other columns sort as before.

The Theme button (top right) cycles Auto, Light, Dark and Eye care, a light theme in a soft cream (#FAF9DE): the window background, the cards, the boxes, the buttons and every row of the lists are cream instead of white or light gray.

**Who has what**

- Open by: which window has the sequence open, by name and process id. It updates every 10 seconds.
- A sequence open in another window cannot be opened, discarded or marked done until it is closed there. A window that crashes releases its sequence at once.
- Your name (in the window title) is recorded on your anchors and review steps: the user you signed in as when this computer runs the CutiePlus service; otherwise `--labeler`, else `CUTIE_LABELER`, else the account name.
- Annotators: everyone who changed the sequence (any change) or marked it done (Mark done, or Ctrl+Q in the labeling window); Save does not count. The Statistics card (bottom right, beside the messages) counts the sequences each annotator worked on in this tab, whatever their status now; a sequence with two annotators counts for both. Type a name in the filter to list that person's sequences.
- Total Intervals (the last line of the Statistics card): the intervals of every sequence marked done, in all tabs, whatever tab is shown. Each interval is a clip of the dataset (Export); a sequence with no interval of its own counts as one (the whole sequence). A done sequence later flagged Need Attention still counts; a discarded one does not.

Outside task mode (a plain Cutie workspace), C commits the frame to permanent memory, Ctrl+S exports binary masks, and the Advanced card keeps the original Cutie's overlay video (Save visualization, Save soft mask, Export as video, Ctrl+E).

**With several people (sign-in)**

When this computer runs the CutiePlus service, every window starts with a sign-in; the first time, use the setup token you were given. Your name and role show beside the theme button, with Log out.
- The GPU is shared. A propagation may wait for others' first: the window says how many are ahead, stays editable meanwhile, and follows the run once it starts; Pause cancels it. GPU queue (task browser) shows the queue and your propagations, with Cancel and Retry.
- A propagation's masks are saved when it ends. If you edited the sequence after asking for it, its result is held: the window asks whether to apply it (anchors are kept) or discard it, now or the next time the sequence opens.
- Supervisors and admins: Manage Annotators (task browser) adds collaborators, their SSH keys and desktops. See docs/COOPERATION.md.
- If the service cannot be reached, you can work offline: browse, edit and save, but no clicks or propagation.
