# Tests (run against a scratch instance, never the live one)

    S=<folder with speech.wav (6 s of Thai speech, 16 kHz mono) and silence.wav>
    mkdir -p $S/scr && echo -n testcode123 > $S/scr/access.token
    ZECRET_DATA=$S/scr ZECRET_PORT=8445 ZECRET_SILENT_S=3 .venv/bin/python app/server.py &
    curl -sk -X POST https://127.0.0.1:8445/api/auth/setup -H 'Content-Type: application/json' -H 'Origin: https://127.0.0.1:8445' \
         -d '{"code":"testcode123","username":"boss","password":"adminpassword123","name":"Boss"}'
    .venv/bin/python -W ignore tools/tests/test_soundcheck_api.py $S        # host sound check endpoint + storage item
    .venv/bin/python -W ignore tools/tests/test_soundcheck_ws.py $S         # recorder watchdog over /ws/record
    .venv/bin/python -W ignore tools/tests/test_soundcheck_ui.py $S $S/speech.wav 9333    # Record tab in headless Chrome (fake microphone)
    .venv/bin/python -W ignore tools/tests/test_soundcheck_ui.py $S $S/silence.wav 9334
    .venv/bin/python -W ignore tools/tests/test_phone_mic.py $S            # join phones: /join/<token>/sound + guest watchdog
    .venv/bin/python -W ignore tools/tests/test_phone_mic_ui.py $S $S/speech.wav 9333     # join.html in headless Chrome
    .venv/bin/python -W ignore tools/tests/test_phone_mic_ui.py $S $S/silence.wav 9334
    .venv/bin/python -W ignore tools/tests/test_mode_choice.py $S $S/speech.wav 9333   # Start asks live captions / record only
    .venv/bin/python -W ignore tools/tests/test_language.py $S              # meeting language th / th-en / en (needs english.wav, mixed.wav, thai.wav)
    .venv/bin/python -W ignore tools/tests/test_email_accounts.py $S/scr    # e-mail To / CC per account (copy a finished meeting folder into $S/scr/meetings first)
    tools/tests/scratch_kill.sh && rm -rf $S/scr                           # stop the scratch instance (by its ZECRET_PORT=8445 environment)

The WAV samples are not in the repository (they come from a real meeting). Run the UI tests one at a time: phone checks are
throttled per device address + name, and the first check after a server start loads the speech model (about 15 s).
