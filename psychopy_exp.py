

"""PsychoPy experiment: Happy vs Sad classification

- Loads stimuli list from brain_study_metadata.csv (created by dataset_creation.py)
- Images live in ./super_final_pictures
- Response keys:
    LEFT arrow  -> happy
    RIGHT arrow -> sad
- Saves RT + correctness + trial metadata for EACH run into a new folder/file under ./results/

Run this with PsychoPy (Python). Recommended: PsychoPy standalone.
"""

from __future__ import annotations

import os
import random
from datetime import datetime

from psychopy import visual, core, event, gui, data


# ----------------------------
# Paths (relative to this file)
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "super_final_pics")
STIM_CSV = os.path.join(SCRIPT_DIR, "brain_study_metadata.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ----------------------------
# Experiment parameters
# ----------------------------
N_TRIALS = 79  # per participant, ~5 minutes with timings below
FIXATION_DUR = 0.50  # seconds (before each trial)
MAX_RESP_TIME = 1.50  # seconds (image stays up until response or timeout)
ITI_DUR = 0.30  # seconds (blank screen after response)

KEY_HAPPY = "right"
KEY_SAD = "left"
QUIT_KEYS = ["escape"]


def show_text(win: visual.Window, text: str, wait_keys: list[str] | None = None) -> None:
    """Display centered text; wait for a key press if wait_keys is provided."""
    stim = visual.TextStim(win, text=text, height=0.06, wrapWidth=1.3, color="white")
    stim.draw()
    win.flip()
    if wait_keys is None:
        return
    event.clearEvents(eventType="keyboard")
    event.waitKeys(keyList=wait_keys)


def draw_fixation(win: visual.Window, dur: float) -> None:
    """Draw a fixation cross for `dur` seconds."""
    cross = visual.TextStim(win, text="+", height=0.12, color="white")
    cross.draw()
    win.flip()
    core.wait(dur)


def load_trials_from_csv(csv_path: str) -> list[dict]:
    """Load rows from CSV created by dataset_creation.py.

    Expects columns: filename, source, gender, emotion
    We keep only emotion in {happy, sad}.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Make sure brain_study_metadata.csv exists."
        )

    rows = []
    # PsychoPy's data.importConditions reads CSV into a list of dicts
    for row in data.importConditions(csv_path):
        # Normalize possible casing/spaces
        emo = str(row.get("emotion", "")).strip().lower()
        if emo not in ("happy", "sad"):
            continue
        # Keep only the fields we care about (plus any extra columns if present)
        row["emotion"] = emo
        rows.append(row)

    if not rows:
        raise ValueError(
            "No usable rows found in CSV. Ensure emotion column contains 'happy' or 'sad'."
        )

    return rows


def pick_practice_examples(all_rows: list[dict]) -> list[dict]:
    """Pick up to 4 practice rows: happy/sad x (ai/non-ai) if possible."""
    def match(emo: str, src: str) -> list[dict]:
        return [r for r in all_rows if r.get("emotion") == emo and str(r.get("source", "")).lower() == src]

    practice = []
    targets = [("happy", "ai"), ("happy", "non-ai"), ("sad", "ai"), ("sad", "non-ai")]
    for emo, src in targets:
        pool = match(emo, src)
        if pool:
            practice.append(random.choice(pool))

    # If we couldn't find all combos, just sample up to 4 from everything
    if len(practice) < 4:
        remaining = [r for r in all_rows if r not in practice]
        random.shuffle(remaining)
        practice.extend(remaining[: max(0, 4 - len(practice))])

    return practice[:4]


def ensure_results_path(participant: str, session: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Save each run in a new subfolder ("new repository")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_participant = "".join(ch for ch in participant if ch.isalnum() or ch in ("-", "_")) or "anon"
    safe_session = "".join(ch for ch in session if ch.isalnum() or ch in ("-", "_")) or "001"

    run_dir = os.path.join(RESULTS_DIR, f"sub-{safe_participant}", f"ses-{safe_session}", run_stamp)
    os.makedirs(run_dir, exist_ok=True)

    # PsychoPy will add .csv/.xlsx/.psydat depending on saveAs...; we’ll save CSV explicitly.
    return os.path.join(run_dir, f"sub-{safe_participant}_ses-{safe_session}_{run_stamp}")


def main() -> None:
    # ----------------------------
    # Participant dialog
    # ----------------------------
    info = {
        "participant": "",  # e.g., 01
        "session": "001",  # optional
        "full_screen": True,
    }
    dlg = gui.DlgFromDict(dictionary=info, title="Happy/Sad Experiment")
    if not dlg.OK:
        return

    participant = str(info["participant"]).strip()
    session = str(info["session"]).strip()
    full_screen = bool(info["full_screen"])

    # ----------------------------
    # Load stimuli
    # ----------------------------
    all_rows = load_trials_from_csv(STIM_CSV)

    # Randomize trials WITHOUT replacement (no image can appear twice)
    trials_rows = all_rows.copy()
    random.shuffle(trials_rows)

    # If you want exactly N_TRIALS, truncate after shuffling
    if len(trials_rows) < N_TRIALS:
        raise ValueError(
            f"Not enough unique images: requested {N_TRIALS}, but only {len(trials_rows)} available."
        )

    trials_rows = trials_rows[:N_TRIALS]

    # Practice examples (4 images)
    practice_rows = pick_practice_examples(all_rows)

    # ----------------------------
    # Setup window
    # ----------------------------
    win = visual.Window(
        size=(1280, 720),
        fullscr=full_screen,
        color="black",
        units="height",
    )

    # Prepare stimuli
    image_stim = visual.ImageStim(win, image=None, size=(0.9, 0.9))

    # ----------------------------
    # Data saving
    # ----------------------------
    base_filename = ensure_results_path(participant, session)
    exp_name = "happy_sad_faces"
    this_exp = data.ExperimentHandler(
        name=exp_name,
        version="1.0",
        extraInfo={"participant": participant, "session": session},
        dataFileName=base_filename,
        savePickle=True,
        saveWideText=False,  # we'll call saveAsWideText at the end
    )

    # ----------------------------
    # Instructions
    # ----------------------------
    instructions = (
        "You will see faces one by one.\n\n"
        f"Press RIGHT arrow for HAPPY\n"
        f"Press LEFT arrow for SAD\n\n"
        "Respond as quickly and accurately as possible.\n\n"
        "Press SPACE to start."
    )
    show_text(win, instructions, wait_keys=["space"])

    # Calibration/fixation cross BEFORE the experiment starts
    draw_fixation(win, dur=1.50)

    # Show 4 example images (1 second each) so participants understand the flow
    if practice_rows:
        show_text(win, "Examples (no responses recorded).\nPress SPACE to continue.", wait_keys=["space"])
        for pr in practice_rows:
            fname = str(pr.get("filename", ""))
            img_path = os.path.join(IMAGE_DIR, fname)
            if os.path.isfile(img_path):
                image_stim.image = img_path
                image_stim.draw()
                win.flip()
                core.wait(1.0)
            # brief blank
            win.flip()
            core.wait(0.3)

    show_text(win, "Experiment starts now.\nPress SPACE.", wait_keys=["space"])

    # ----------------------------
    # Trial loop
    # ----------------------------
    kb_clock = core.Clock()

    for t_idx, row in enumerate(trials_rows, start=1):
        # Allow immediate quit
        if event.getKeys(keyList=QUIT_KEYS):
            break

        # Fixation before each trial
        draw_fixation(win, FIXATION_DUR)

        # Prepare image
        filename = str(row.get("filename", ""))
        img_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.isfile(img_path):
            # Skip missing images but log it
            this_exp.addData("trial_index", t_idx)
            this_exp.addData("filename", filename)
            this_exp.addData("missing_file", True)
            this_exp.addData("resp_key", "")
            this_exp.addData("rt", "")
            this_exp.addData("correct", "")
            this_exp.nextEntry()
            continue

        correct_emotion = str(row.get("emotion", "")).strip().lower()

        # Draw stimulus and collect response
        event.clearEvents(eventType="keyboard")
        kb_clock.reset()

        image_stim.image = img_path
        image_stim.draw()
        win.flip()

        keys = event.waitKeys(
            maxWait=MAX_RESP_TIME,
            keyList=[KEY_HAPPY, KEY_SAD] + QUIT_KEYS,
            timeStamped=kb_clock,
        )

        # Default values
        resp_key = ""
        rt = ""
        correct = ""

        if keys:
            key, t = keys[0]
            if key in QUIT_KEYS:
                break
            resp_key = key
            rt = float(t)

            # Determine correctness
            if correct_emotion == "happy":
                correct = (resp_key == KEY_HAPPY)
            elif correct_emotion == "sad":
                correct = (resp_key == KEY_SAD)
            else:
                correct = ""
        else:
            # No response within MAX_RESP_TIME
            resp_key = ""
            rt = ""
            correct = False  # treat timeout as incorrect

        # Log trial data
        this_exp.addData("trial_index", t_idx)
        this_exp.addData("filename", filename)
        this_exp.addData("source", row.get("source", ""))
        this_exp.addData("gender", row.get("gender", ""))
        this_exp.addData("emotion", correct_emotion)
        this_exp.addData("resp_key", resp_key)
        this_exp.addData("rt", rt)
        this_exp.addData("correct", correct)
        this_exp.addData("missing_file", False)
        this_exp.nextEntry()

        # ITI (blank)
        win.flip()
        core.wait(ITI_DUR)

    # ----------------------------
    # End / save
    # ----------------------------
    show_text(win, "Thank you!\n\nPress SPACE to finish.", wait_keys=["space"])

    # Save a wide CSV for easy analysis
    this_exp.saveAsWideText(base_filename + ".csv")
    this_exp.abort()  # closes handler cleanly

    win.close()
    core.quit()


if __name__ == "__main__":
    main()
