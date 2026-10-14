"""Enrolment guide: a small agent that explains and guides participants through voice enrolment, and tells them the address.

It is deliberately narrow and rule-based (no language model): answers come from a fixed bilingual FAQ filled with the real address,
time left and Wi-Fi note, so it is instant, uses no memory, and cannot invent anything. Questions outside enrolment get a polite
"I can only help with enrolment" and a few example questions.
"""
from __future__ import annotations

import re

STEPS = [
    {"th": ("ต่อ Wi-Fi เดียวกัน", "ใช้ Wi-Fi หรือเครือข่ายเดียวกับห้องประชุม{wifi_th}"), "en": ("Join the same Wi-Fi", "Use the same Wi-Fi / network as the meeting room{wifi_en}")},
    {"th": ("สแกน QR หรือพิมพ์ที่อยู่", "สแกน QR บนจอ หรือพิมพ์ {link} ในช่องที่อยู่ของเบราว์เซอร์"), "en": ("Scan the QR or type the address", "Scan the QR on the screen, or type {link} in the browser's address bar")},
    {"th": ("ใส่ชื่อและยินยอม", "พิมพ์ชื่อของคุณ แล้วติ๊กช่องยินยอม"), "en": ("Type your name and agree", "Type your name and tick the consent box")},
    {"th": ("อนุญาตไมโครโฟน", "เมื่อเบราว์เซอร์ถาม ให้กด “อนุญาต”"), "en": ("Allow the microphone", "When the browser asks, tap “Allow”")},
    {"th": ("อ่านข้อความให้ครบ", "อ่านข้อความบนหน้าจอด้วยเสียงปกติประมาณ 30 วินาที แล้วกด “หยุดและส่ง”"), "en": ("Read the text aloud", "Read the text on the screen in your normal voice for about 30 seconds, then tap “Stop and send”")},
    {"th": ("รอการอนุมัติ", "เจ้าของห้องจะตรวจและกดอนุมัติ หลังจากนั้นระบบจะรู้จักเสียงของคุณ"), "en": ("Wait for approval", "The host checks and approves it; after that the system recognises your voice")},
]

ROTATE = [  # short messages the display screen cycles through
    {"th": "สวัสดีครับ ผมคือ{guide} ผู้ช่วยแนะนำการลงทะเบียนเสียง", "en": "Hello, I am {guide}, your voice-enrolment guide"},
    {"th": "เปิดเบราว์เซอร์แล้วสแกน QR หรือพิมพ์ที่อยู่: {link}", "en": "Open your browser and scan the QR, or type: {link}"},
    {"th": "ที่อยู่ของระบบนี้คือ {app} (ต้องต่อ Wi-Fi เดียวกัน){wifi_th}", "en": "This system's address is {app} (same Wi-Fi needed){wifi_en}"},
    {"th": "ใช้เวลาประมาณ 30 วินาที: อ่านข้อความที่แสดงบนหน้าจอโทรศัพท์ของคุณ", "en": "It takes about 30 seconds: read the text shown on your phone"},
    {"th": "ไม่เก็บไฟล์เสียงของคุณ เก็บเฉพาะ “ลายเสียง” และเจ้าของห้องต้องอนุมัติก่อนใช้งาน", "en": "Your recording is not kept: only a voiceprint, and the host must approve it first"},
    {"th": "ถ้าเบราว์เซอร์เตือนว่า “ไม่ปลอดภัย” ให้กด ขั้นสูง → ดำเนินการต่อ (เครื่องนี้ใช้ใบรับรองที่สร้างเอง)", "en": "If the browser warns “not secure”, tap Advanced → Continue (this device uses its own certificate)"},
    {"th": "ลิงก์เปิดอยู่อีก {mins} นาที", "en": "The link stays open for another {mins} minutes"},
]

# id, Thai keys, English keys, Thai question (suggestion chip), English question, Thai answer, English answer
FAQ = [
    {"id": "greet", "k_th": ["สวัสดี", "หวัดดี", "ดีครับ", "ดีค่ะ"], "k_en": ["hello", "hi", "hey", "good morning"], "q_th": "สวัสดี", "q_en": "Hello",
     "a_th": "สวัสดีครับ ผมคือ{guide} ผมช่วยอธิบายและแนะนำการลงทะเบียนเสียง รวมถึงบอกที่อยู่สำหรับเข้าใช้งานครับ ถามได้เลย เช่น “ต้องทำอย่างไร” หรือ “เปิดที่ไหน”",
     "a_en": "Hello, I am {guide}. I explain and guide voice enrolment and tell you the address to open. Ask me anything about it, for example “what do I do?” or “where do I open it?”"},
    {"id": "who", "k_th": ["คุณคือใคร", "เป็นใคร", "ช่วยอะไรได้", "ทำอะไรได้"], "k_en": ["who are you", "what can you do", "what are you"], "q_th": "คุณช่วยอะไรได้บ้าง", "q_en": "What can you help with?",
     "a_th": "ผมคือ{guide} ผู้ช่วยแนะนำการลงทะเบียนเสียงเท่านั้น: อธิบายขั้นตอน บอกที่อยู่ ช่วยแก้ปัญหาเรื่องไมโครโฟนและคำเตือนของเบราว์เซอร์ และตอบเรื่องความเป็นส่วนตัวครับ",
     "a_en": "I am {guide}, a guide for voice enrolment only: I explain the steps, tell you the address, help with microphone problems and browser warnings, and answer privacy questions."},
    {"id": "steps", "k_th": ["ต้องทำอย่างไร", "ทำยังไง", "ขั้นตอน", "เริ่มยังไง", "เริ่มต้น", "ทำอะไรบ้าง", "วิธี"], "k_en": ["what do i do", "how do i", "steps", "how does it work", "how to", "get started", "instructions"],
     "q_th": "ต้องทำอย่างไรบ้าง", "q_en": "What do I need to do?",
     "a_th": "ขั้นตอนมี 6 ข้อครับ: 1) ต่อ Wi-Fi เดียวกับห้องประชุม 2) สแกน QR หรือพิมพ์ {link} 3) ใส่ชื่อและติ๊กยินยอม 4) กดอนุญาตไมโครโฟน 5) อ่านข้อความบนหน้าจอประมาณ 30 วินาที แล้วกด “หยุดและส่ง” 6) รอเจ้าของห้องอนุมัติ",
     "a_en": "There are 6 steps: 1) join the meeting room's Wi-Fi 2) scan the QR or type {link} 3) type your name and tick consent 4) allow the microphone 5) read the text on screen for about 30 seconds, then tap “Stop and send” 6) wait for the host to approve."},
    {"id": "address", "k_th": ["ที่อยู่", "ไอพี", "ไอพีแอดเดรส", "ip", "ลิงก์", "ลิงค์", "เปิดที่ไหน", "เข้าที่ไหน", "เข้ายังไง", "เข้าอย่างไร", "เปิดยังไง", "พิมพ์ที่ไหน", "url", "เว็บ", "คิวอาร์", "qr", "สแกน", "พอร์ต"],
     "k_en": ["address", "ip", "link", "url", "where do i", "where can i", "how do i open", "how do i get", "open the", "qr", "website", "type", "port", "scan"], "q_th": "เปิดที่ไหน / ที่อยู่คืออะไร", "q_en": "Where do I open it? What is the address?",
     "a_th": "เปิดเบราว์เซอร์ในโทรศัพท์หรือคอมพิวเตอร์ที่ต่อ Wi-Fi/เครือข่ายเดียวกับห้องประชุม แล้วสแกน QR บนจอ หรือพิมพ์ที่อยู่นี้ในช่องที่อยู่ของเบราว์เซอร์ (ไม่ใช่ช่องค้นหา):\n{link}\nที่อยู่ของระบบนี้คือ {app}\nถ้าเปิดไม่ได้ ให้ตรวจว่าต่อ Wi-Fi เดียวกัน{wifi_th}",
     "a_en": "Open a browser on a phone or computer that is on the same Wi-Fi / network as the meeting room, then scan the QR on the screen or type this in the browser's address bar (not the search box):\n{link}\nThe system's address is {app}\nIf it does not open, check that you are on the same Wi-Fi{wifi_en}"},
    {"id": "wifi", "k_th": ["wi-fi", "wifi", "ไวไฟ", "ไวฟาย", "เครือข่าย", "อินเทอร์เน็ต", "ออนไลน์", "เน็ต"], "k_en": ["wi-fi", "wifi", "network", "internet", "same network", "online", "connect"], "q_th": "ต้องต่อ Wi-Fi อะไร", "q_en": "Which Wi-Fi do I need?",
     "a_th": "ต้องต่อ Wi-Fi หรือเครือข่ายเดียวกับเครื่องในห้องประชุมครับ (ไม่ต้องใช้อินเทอร์เน็ต){wifi_th}", "a_en": "You need the same Wi-Fi / network as the meeting-room device (no internet is needed){wifi_en}"},
    {"id": "what", "k_th": ["ทำไมต้อง", "คืออะไร", "ลงทะเบียนเสียง", "ลายเสียง", "เพื่ออะไร", "ใช้ทำอะไร"], "k_en": ["why", "what is", "what for", "voiceprint", "purpose", "enrol", "enroll"], "q_th": "ทำไมต้องลงทะเบียนเสียง", "q_en": "Why do I need to enrol my voice?",
     "a_th": "เพื่อให้ระบบรู้ว่าใครพูดอะไรในการประชุม แล้วใส่ชื่อของคุณหน้าข้อความที่คุณพูดในบันทึกและรายงาน ระบบสร้าง “ลายเสียง” (ค่าตัวเลขที่อธิบายเสียงของคุณ) จากเสียงที่คุณอ่าน",
     "a_en": "So the system knows who said what in the meeting and puts your name next to your words in the minutes and report. It makes a “voiceprint” (numbers describing your voice) from what you read aloud."},
    {"id": "time", "k_th": ["นานแค่ไหน", "กี่วินาที", "กี่นาที", "ใช้เวลา", "นานไหม", "เร็วไหม"], "k_en": ["how long", "how many seconds", "takes", "quick", "time does it"], "q_th": "ใช้เวลานานแค่ไหน", "q_en": "How long does it take?",
     "a_th": "ประมาณ 30 วินาทีครับ ระบบต้องได้ยินเสียงพูดจริงอย่างน้อย 8–10 วินาที ถ้าพูดสั้นไประบบจะบอกให้ลองใหม่", "a_en": "About 30 seconds. The system needs at least 8–10 seconds of real speech; if it is too short it will ask you to try again."},
    {"id": "read", "k_th": ["อ่านอะไร", "พูดอะไร", "อ่านยังไง", "ข้อความ", "อ่านข้อความ", "ต้องพูด"], "k_en": ["what do i read", "what do i say", "read", "say"], "q_th": "ต้องอ่านอะไร", "q_en": "What do I read?",
     "a_th": "อ่านข้อความภาษาไทยที่แสดงบนหน้าจอ ด้วยเสียงปกติ ถือโทรศัพท์ห่างปากประมาณ 20–30 ซม. และอ่านให้ครบทั้งย่อหน้า ไม่ต้องรีบหรือตะโกน",
     "a_en": "Read the Thai text shown on the screen in your normal voice. Hold the phone about 20–30 cm from your mouth and read the whole paragraph; no need to rush or shout."},
    {"id": "audio", "k_th": ["เก็บเสียง", "ไฟล์เสียง", "อัดเสียง", "บันทึกเสียง", "เก็บไว้ไหม", "ส่วนตัว", "ความเป็นส่วนตัว", "ปลอดภัย", "ข้อมูล"], "k_en": ["recording kept", "keep my voice", "stored", "privacy", "private", "safe", "saved", "my data"], "q_th": "เก็บไฟล์เสียงของฉันไหม", "q_en": "Is my recording kept?",
     "a_th": "ไม่เก็บไฟล์เสียงครับ ระบบเก็บเฉพาะ “ลายเสียง” (ตัวเลข) ไว้ในเครื่องในห้องประชุมนี้เท่านั้น ไม่ส่งออกอินเทอร์เน็ต และเจ้าของห้องต้องอนุมัติก่อนนำไปใช้",
     "a_en": "No. Your recording is not kept: only the voiceprint (numbers) is stored on the meeting-room device. It does not go to the internet, and the host must approve it before it is used."},
    {"id": "delete", "k_th": ["ลบ", "ถอนความยินยอม", "ยกเลิก", "เอาออก"], "k_en": ["delete", "remove", "withdraw", "cancel", "forget me"], "q_th": "ขอลบเสียงของฉันได้ไหม", "q_en": "Can I have my voice deleted?",
     "a_th": "ได้ครับ แจ้งเจ้าของห้อง แล้วเขาลบโปรไฟล์เสียงของคุณได้ทุกเมื่อในหน้า “เสียง” ของระบบ", "a_en": "Yes. Tell the host; they can delete your voice profile at any time in the system's Voices page."},
    {"id": "approval", "k_th": ["อนุมัติ", "รอ", "เจ้าของห้อง", "โฮสต์", "ผู้จัด"], "k_en": ["approve", "approval", "wait", "host", "organiser", "organizer"], "q_th": "ทำไมต้องรอการอนุมัติ", "q_en": "Why does the host have to approve?",
     "a_th": "เพื่อป้องกันไม่ให้ใครลงทะเบียนโดยใช้ชื่อของคนอื่น เจ้าของห้องจะตรวจชื่อและกดอนุมัติ หลังอนุมัติระบบจะรู้จักเสียงของคุณในการประชุมครั้งต่อไปด้วย",
     "a_en": "So nobody can enrol under someone else's name. The host checks the name and approves it; after that the system also recognises you in future meetings."},
    {"id": "cert", "k_th": ["ไม่ปลอดภัย", "คำเตือน", "ใบรับรอง", "ไม่เป็นส่วนตัว", "ขั้นสูง", "ดำเนินการต่อ", "ssl", "https", "แดง", "หน้าแดง"], "k_en": ["not secure", "warning", "certificate", "privacy error", "advanced", "proceed", "unsafe", "dangerous", "connection is not private", "ssl", "https"],
     "q_th": "เบราว์เซอร์เตือนว่าไม่ปลอดภัย", "q_en": "The browser says it is not secure",
     "a_th": "เป็นเรื่องปกติครับ เพราะเครื่องในห้องประชุมใช้ใบรับรองที่สร้างเอง และอยู่ในเครือข่ายภายใน ไปต่อได้ถ้าที่อยู่ตรงกับ {app}\n• Android/Chrome: กด “ขั้นสูง” → “ไปที่ … (ไม่ปลอดภัย)”\n• iPhone/Safari: กด “แสดงรายละเอียด” → “เข้าชมเว็บไซต์นี้”",
     "a_en": "That is normal: the meeting-room device uses its own certificate and sits on a local network. It is fine to continue if the address matches {app}\n• Android/Chrome: tap “Advanced” → “Proceed to … (unsafe)”\n• iPhone/Safari: tap “Show details” → “visit this website”"},
    {"id": "mic", "k_th": ["ไมโครโฟน", "ไมค์", "ไมโครโฟน", "ไม่ได้ยิน", "ไม่ขอสิทธิ์", "บล็อก", "ถูกบล็อก", "อนุญาต", "ใช้ไม่ได้", "ขึ้นว่า"], "k_en": ["microphone", "mic", "blocked", "permission", "allow", "no sound", "not working", "denied"], "q_th": "ใช้ไมโครโฟนไม่ได้", "q_en": "The microphone does not work",
     "a_th": "กด “อนุญาต” เมื่อเบราว์เซอร์ถามเรื่องไมโครโฟน ถ้าเผลอกดห้ามไปแล้ว: แตะไอคอนด้านซ้ายของแถบที่อยู่ (หรือ ตั้งค่าเว็บไซต์) → ไมโครโฟน → อนุญาต แล้วรีโหลดหน้านี้ และตรวจว่าไม่มีแอปอื่นใช้ไมโครโฟนอยู่",
     "a_en": "Tap “Allow” when the browser asks about the microphone. If you blocked it: tap the icon at the left of the address bar (or Site settings) → Microphone → Allow, then reload this page, and make sure no other app is using the microphone."},
    {"id": "quiet", "k_th": ["เสียงเบา", "เบา", "ดังไป", "รบกวน", "เสียงดัง", "เงียบ", "ไม่พอ", "สั้นไป", "พูดไม่ถึง", "ไม่ผ่าน", "ลองใหม่"], "k_en": ["quiet", "too short", "not enough", "noise", "noisy", "loud", "try again", "failed", "rejected"], "q_th": "ระบบบอกว่าเสียงไม่พอ", "q_en": "It says not enough speech",
     "a_th": "ลองใหม่ในที่เงียบ ถือโทรศัพท์ห่างปากประมาณ 20–30 ซม. พูดด้วยเสียงปกติให้ดังพอ และอ่านข้อความให้ครบ (ประมาณ 30 วินาที) ดูแถบสีเขียวขยับเมื่อคุณพูดหรือไม่ ถ้าไม่ขยับแปลว่าไมโครโฟนยังไม่ทำงาน",
     "a_en": "Try again somewhere quiet. Hold the phone 20–30 cm from your mouth, speak at a normal, clear volume and read the whole text (about 30 seconds). Watch the green bar: if it does not move when you talk, the microphone is not working."},
    {"id": "name", "k_th": ["ชื่ออะไร", "ใส่ชื่อ", "ชื่อเล่น", "ชื่อจริง", "นามสกุล"], "k_en": ["what name", "my name", "nickname", "surname", "which name"], "q_th": "ควรใส่ชื่ออะไร", "q_en": "Which name should I type?",
     "a_th": "ใส่ชื่อที่ทุกคนในห้องเรียกคุณ (ชื่อเล่นหรือชื่อจริงก็ได้) ถ้ามีรายชื่อผู้เข้าร่วมอยู่แล้ว ให้พิมพ์ให้ตรงกัน เพื่อให้รายงานจับคู่ชื่อได้ถูกต้อง",
     "a_en": "Type the name people in the room call you (nickname or real name). If there is a participant list, use the same spelling so the report can match you."},
    {"id": "other", "k_th": ["คนอื่น", "แทน", "อีกคน", "หลายคน", "ลงทะเบียนให้"], "k_en": ["someone else", "another person", "for others", "my colleague", "friend", "enrol another", "enroll another"], "q_th": "ลงทะเบียนให้คนอื่นได้ไหม", "q_en": "Can I enrol someone else?",
     "a_th": "แต่ละคนต้องอ่านด้วยเสียงของตัวเองและยินยอมเอง ง่ายที่สุดคือให้ทุกคนทำบนโทรศัพท์ของตัวเอง หรือส่งโทรศัพท์ต่อให้เจ้าตัวแล้วกด “ลงทะเบียนอีกคน” หลังส่งเสร็จ",
     "a_en": "Each person must read in their own voice and give their own consent. Easiest is for everyone to use their own phone, or pass the phone to them and tap “Enroll another person” after sending."},
    {"id": "expired", "k_th": ["หมดอายุ", "ใช้ไม่ได้", "ปิดแล้ว", "ลิงก์เสีย", "ลิงก์ปิด", "ไม่เปิด", "เข้าไม่ได้"], "k_en": ["expired", "closed", "not open", "invalid", "link does not work", "can't open", "cannot open"], "q_th": "ลิงก์หมดอายุ / เข้าไม่ได้", "q_en": "The link expired / does not open",
     "a_th": "ลิงก์เปิดได้ช่วงสั้นๆ และเปลี่ยนใหม่ทุกครั้งครับ ขอให้เจ้าของห้องเปิดลิงก์ใหม่ แล้วสแกน QR ใหม่ ถ้ายังเข้าไม่ได้ ให้ตรวจว่าต่อ Wi-Fi เดียวกัน{wifi_th}",
     "a_en": "The link is open only for a short time and a new one is made each time. Ask the host to open a new link and scan the new QR. If it still does not open, check you are on the same Wi-Fi{wifi_en}"},
    {"id": "next", "k_th": ["หลังจากนั้น", "เสร็จแล้ว", "ส่งแล้ว", "ต่อไป", "ถัดไป", "หลังส่ง"], "k_en": ["what next", "after", "done", "finished", "sent"], "q_th": "ส่งแล้วต้องทำอะไรต่อ", "q_en": "What happens after I send it?",
     "a_th": "ไม่ต้องทำอะไรต่อครับ เจ้าของห้องจะเห็นรายการของคุณและกดอนุมัติ หลังจากนั้นระบบจะใส่ชื่อคุณเมื่อได้ยินเสียงคุณในการประชุมครับ", "a_en": "Nothing more to do. The host will see your entry and approve it; after that the system puts your name on what it hears you say in meetings."},
    {"id": "lang", "k_th": ["ภาษาอังกฤษ", "ภาษาไทย", "ต้องพูดไทย", "พูดภาษา"], "k_en": ["english", "thai", "language", "speak thai"], "q_th": "ต้องพูดภาษาไทยไหม", "q_en": "Do I have to read in Thai?",
     "a_th": "แนะนำให้อ่านข้อความภาษาไทยที่แสดงบนหน้าจอ เพราะเป็นภาษาที่คุณจะใช้ในการประชุม ระบบสร้างลายเสียงจากลักษณะเสียงของคุณ", "a_en": "Please read the Thai text shown, since that is the language you will use in the meeting. The system builds the voiceprint from the character of your voice."},
]
KEYS = {  # id -> (Thai keys, English keys); (phrase, weight) forces a weight for short domain words
    "greet": (["สวัสดี", "หวัดดี"], [("hello", 6), ("hi", 6), ("hey", 6), "good morning"]),
    "who": (["คุณคือใคร", "เป็นใคร", "ช่วยอะไรได้", "ทำอะไรได้"], ["who are you", "what can you do", "what are you"]),
    "steps": (["ต้องทำอย่างไร", "ทำยังไง", "ขั้นตอน", "เริ่มยังไง", "ทำอะไรบ้าง", "ต้องทำอะไร", "วิธีลงทะเบียน", "เริ่มตรงไหน"], ["what do i do", "how do i do", "how do i start", "how does it work", "steps", "instructions", "how to enrol", "how to enroll", "get started"]),
    "address": (["ที่อยู่", ("ไอพี", 5), ("ip", 5), "ลิงก์", "ลิงค์", "เปิดที่ไหน", "เข้าที่ไหน", "เข้ายังไง", "เข้าอย่างไร", "เปิดยังไง", "พิมพ์ที่ไหน", "คิวอาร์", ("qr", 5), "สแกน", "พอร์ต", "เว็บ"],
                ["address", ("ip", 5), "the link", "url", "where do i", "where can i", "how do i open", "how do i get to", "website", "port", ("qr", 5), "scan"]),
    "wifi": ([("ไวไฟ", 5), ("wifi", 5), "wi-fi", "เครือข่าย", "ไวฟาย"], [("wifi", 5), "wi-fi", "same network", "which network", "internet"]),
    "what": (["ลงทะเบียนเสียง", "ลายเสียง", "ทำไมต้องลงทะเบียน", "ลงทะเบียนทำไม"], ["voiceprint", "why enrol", "why enroll", "why do i need to enrol", "why do i need to enroll", "need to enrol", "need to enroll", "purpose of enrol"]),
    "time": (["นานแค่ไหน", "กี่วินาที", "กี่นาที", "ใช้เวลา", "นานไหม", "เร็วไหม"], ["how long", "how many seconds", "how many minutes", "how quick"]),
    "read": (["อ่านอะไร", "พูดอะไร", "อ่านยังไง", "อ่านข้อความ", "ต้องพูดอะไร"], ["what do i read", "what do i say", "what should i read", "read aloud", "what to read"]),
    "audio": (["เก็บเสียง", "ไฟล์เสียง", "อัดเสียง", "บันทึกเสียง", "เก็บไว้ไหม", "ความเป็นส่วนตัว", "ข้อมูลส่วนตัว", "ปลอดภัยไหม"], ["recording kept", "keep my voice", "keep my recording", "privacy", "is it stored", "my data", "is it safe"]),
    "delete": (["ลบเสียง", "ลบโปรไฟล์", "ถอนความยินยอม", "ขอลบ", "ยกเลิกความยินยอม"], ["delete my voice", "delete my", "remove my voice", "withdraw", "forget me", "delete it later", "delete my voice later"]),
    "approval": (["อนุมัติ", "เจ้าของห้อง", "รอเจ้าของ"], ["approve", "approval", "host need"]),
    "cert": (["ไม่ปลอดภัย", "คำเตือน", "ใบรับรอง", "ขั้นสูง", "ดำเนินการต่อ", ("ssl", 5), ("https", 5), "หน้าแดง"], ["not secure", "warning", "certificate", "privacy error", "advanced", "proceed", "unsafe", "not private", ("ssl", 5), ("https", 5)]),
    "mic": (["ไมโครโฟน", ("ไมค์", 5), "ไม่ได้ยิน", "ไม่ขอสิทธิ์", "ถูกบล็อก", "ขอสิทธิ์"], ["microphone", ("mic", 5), "blocked", "permission", "no sound", "denied"]),
    "quiet": (["เสียงเบา", "เสียงดังไป", "เสียงรบกวน", "สั้นไป", "พูดไม่ถึง", "เสียงไม่พอ", "ไม่พอ", "ลองใหม่"], ["quiet", "too short", "not enough speech", "not enough", "noisy", "try again", "failed"]),
    "name": (["ชื่ออะไร", "ใส่ชื่อ", "ชื่อเล่น", "ชื่อจริง"], ["what name", "which name", "my name", "nickname"]),
    "other": (["คนอื่น", "เพื่อน", "อีกคน", "ลงทะเบียนให้", "แทนคน"], ["someone else", "another person", "my colleague", "my friend", "enrol another", "enroll another", "enrol someone", "enroll someone", "for others"]),
    "expired": (["หมดอายุ", "ลิงก์เสีย", "ลิงก์ปิด", "ไม่เปิด", "เข้าไม่ได้", "ลิงก์ใช้ไม่ได้"], ["expired", "link does not work", "not open", "cannot open", "can't open", "link closed"]),
    "next": (["ส่งแล้วต้อง", "หลังส่ง", "เสร็จแล้วต้อง", "ต้องทำอะไรต่อ"], ["what next", "what happens after", "after i send", "after sending", "what happens next"]),
    "lang": (["ภาษาอังกฤษ", "ต้องพูดไทย", "ต้องอ่านภาษา", "ภาษาไทยไหม"], ["in english", "speak thai", "read in thai", "which language"]),
    "thanks": (["ขอบคุณ"], ["thank you", "thanks"]),
}
FAQ.append({"id": "thanks", "k_th": [], "k_en": [], "q_th": "ขอบคุณ", "q_en": "Thanks", "a_th": "ยินดีครับ มีอะไรอยากถามเรื่องการลงทะเบียนเสียงอีกไหมครับ", "a_en": "You are welcome. Anything else about voice enrolment?"})
for _e in FAQ:
    _e["k_th"], _e["k_en"] = KEYS[_e["id"]]

OUT_OF_SCOPE = {"th": "ผมช่วยตอบเรื่องการลงทะเบียนเสียงและที่อยู่สำหรับเข้าใช้งานเท่านั้นครับ ลองถามเช่น: “ต้องทำอย่างไร” “เปิดที่ไหน” “เก็บเสียงไหม”",
                "en": "I can only help with voice enrolment and the address to open. Try asking: “what do I do?”, “where do I open it?”, “is my voice kept?”"}


def lang_of(text: str) -> str:
    return "th" if re.search(r"[฀-๿]", text or "") else "en"


def context(app_urls: list[str], enroll_urls: list[str], minutes_left: int, wifi: str, guide: str) -> dict:
    wifi = " ".join((wifi or "").split())[:200]
    return {"app": app_urls[0] if app_urls else "(ที่อยู่ของเครื่องนี้ / this device's address)", "link": enroll_urls[0] if enroll_urls else "(ลิงก์จากเจ้าของห้อง / the link from the host)",
            "mins": max(0, minutes_left), "guide": guide, "wifi_th": f" — Wi-Fi: {wifi}" if wifi else "", "wifi_en": f" — Wi-Fi: {wifi}" if wifi else ""}


def fill(text: str, c: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(c.get(m.group(1), m.group(0))), text)


STRONG = 5.0  # a match must be at least this specific (matched characters, or an explicit weight) to count as an enrolment question


def _score(q: str, entry: dict, lang: str) -> float:
    """Sum of the lengths of the distinct key phrases found: long, specific phrases outweigh short generic ones."""
    ql = q.lower()
    qn = re.sub(r"\s+", "", ql)
    s = 0.0
    for k in entry["k_th"]:
        w, key = (k[1], k[0]) if isinstance(k, tuple) else (len(k), k)
        if re.sub(r"\s+", "", key.lower()) in qn:
            s += w
    for k in entry["k_en"]:
        w, key = (k[1], k[0]) if isinstance(k, tuple) else (len(k), k)
        if re.search(r"(?<![a-z0-9])" + re.escape(key.lower()) + r"(?![a-z0-9])", ql):
            s += w
    return s


def answer(question: str, c: dict, lang: str | None = None) -> dict:
    """-> {"answer": str, "id": faq id | None, "lang": "th"|"en", "suggestions": [questions]}"""
    q = " ".join((question or "").split())[:300]
    lang = lang or lang_of(q)
    scored = sorted(((_score(q, e, lang), i) for i, e in enumerate(FAQ)), key=lambda t: (-t[0], t[1]))
    best, idx = scored[0]
    sugg = [e["q_" + lang] for e in FAQ if e["id"] in ("steps", "address", "audio", "time")]
    if best < STRONG:
        return {"answer": fill(OUT_OF_SCOPE[lang], c), "id": None, "lang": lang, "suggestions": sugg}
    e = FAQ[idx]
    return {"answer": fill(e["a_" + lang], c), "id": e["id"], "lang": lang, "suggestions": [x["q_" + lang] for x in FAQ if x["id"] in ("steps", "address", "audio", "cert") and x["id"] != e["id"]][:3]}


def spoken_address(url: str, lang: str) -> str:
    """The address as words, digit by digit (for the optional read-aloud button)."""
    m = re.match(r"https?://([^/:]+)(?::(\d+))?", url or "")
    if not m:
        return ""
    th = dict(zip("0123456789", "ศูนย์ หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า".split()))
    en = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))
    d = th if lang == "th" else en
    words = []
    for part in m.group(1).split("."):
        words.append(" ".join(d[ch] for ch in part if ch in d) or part)
    host = (" จุด " if lang == "th" else " dot ").join(words)
    port = (" พอร์ต " + " ".join(d[ch] for ch in m.group(2))) if (m.group(2) and lang == "th") else ((" port " + " ".join(d[ch] for ch in m.group(2))) if m.group(2) else "")
    return host + port


def payload(app_urls: list[str], enroll_urls: list[str], minutes_left: int, settings: dict) -> dict:
    """Everything a page needs to show the guide in both languages."""
    c = context(app_urls, enroll_urls, minutes_left, settings.get("guide_wifi", ""), settings.get("guide_name", "น้องแนะนำ"))
    return {"name": c["guide"], "lang_pref": settings.get("guide_lang", "auto"), "app_urls": app_urls, "enroll_urls": enroll_urls, "minutes_left": c["mins"], "wifi": settings.get("guide_wifi", ""),
            "steps": [{lg: [fill(x, c) for x in s[lg]] for lg in ("th", "en")} for s in STEPS], "rotate": [{lg: fill(r[lg], c) for lg in ("th", "en")} for r in ROTATE],
            "suggest": {lg: [e["q_" + lg] for e in FAQ if e["id"] in ("steps", "address", "audio", "time", "cert", "mic")] for lg in ("th", "en")},
            "spoken": {lg: spoken_address(app_urls[0] if app_urls else "", lg) for lg in ("th", "en")}}
