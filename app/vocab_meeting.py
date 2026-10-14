"""Starter pack 'Meeting': the ~650 words that come up most in Thai business meetings (board, management, shareholder,
project and team meetings). Format of each line: term | Thai spellings a recogniser may produce (/-separated) | category | note.
Most important first: the summary prompt only takes the first terms of a pack. Plain Thai words carry no spellings: they
are the correct written form (for the summary and for review), not repair rules. Every Thai spelling was checked against a
62k-word Thai dictionary and real meeting text so that it cannot rewrite an ordinary Thai word."""

TABLE = """
Agenda|อะเจนด้า/อเจนดา/อะเจนดา/เอเจนด้า|term|วาระการประชุม
Minutes|มินิทส์/มินิตส์|term|รายงานการประชุม
Action Item|แอคชั่นไอเท็ม/แอ็กชันไอเท็ม/แอคชั่นไอเทม|term|งานที่ต้องทำ
Follow-up|ฟอลโลว์อัพ/ฟอลโล่อัพ/ฟอลโลอัพ|term|ติดตามผล
Deadline|เดดไลน์/เดทไลน์|term|กำหนดส่ง
Timeline|ไทม์ไลน์|term|
Milestone|ไมล์สโตน|term|
Roadmap|โรดแมป/โรดแมพ|term|
KPI|เคพีไอ|acronym|Key Performance Indicator
OKR|โอเคอาร์|acronym|Objectives and Key Results
Budget|บัดเจ็ท/บัดเจ็ต|term|งบประมาณ
Forecast|ฟอร์แคสต์/ฟอร์แคสท์/โฟร์แคสต์|term|การคาดการณ์
Target|ทาร์เก็ต/ทาร์เกต|term|เป้าหมาย
Strategy|สแตรทิจี/สเตรทเตอจี้/สแตรทเทจี|term|กลยุทธ์
Proposal|โพรโพซัล/โปรโพซัล/โพรโพสอล|term|ข้อเสนอ
Presentation|เพรสเซนเทชั่น/พรีเซนเทชั่น/พรีเซนเทชัน|term|การนำเสนอ
Report|รีพอร์ต/รีพอร์ท|term|รายงาน
Dashboard|แดชบอร์ด/แดชบอร์ท|term|
Feedback|ฟีดแบ็ก/ฟีดแบ็ค/ฟีดแบค|term|ความคิดเห็น
Update|อัปเดต/อัพเดท/อัพเดต/อัปเดท|term|
Kick-off|คิกออฟ/คิกอ็อฟ/คิ๊กออฟ|term|การประชุมเริ่มโครงการ
Workshop|เวิร์กช็อป/เวิร์คช็อป/เวิร์คชอป|term|
Brainstorm|เบรนสตอร์ม/เบรนสตรอม|term|ระดมความคิด
Q&A|คิวแอนด์เอ|term|ถาม-ตอบ
AGM|เอจีเอ็ม|acronym|Annual General Meeting, ประชุมสามัญผู้ถือหุ้นประจำปี
EGM|อีจีเอ็ม|acronym|Extraordinary General Meeting, ประชุมวิสามัญผู้ถือหุ้น
E-AGM|อีเอจีเอ็ม|acronym|ประชุมผู้ถือหุ้นผ่านสื่ออิเล็กทรอนิกส์
E-Meeting|อีมีตติ้ง/อีมีทติ้ง|term|การประชุมผ่านสื่ออิเล็กทรอนิกส์
Hybrid Meeting|ไฮบริดมีตติ้ง/ไฮบริดมีทติ้ง|term|
Online Meeting|ออนไลน์มีตติ้ง/ออนไลน์มีทติ้ง|term|
Meeting|มีตติ้ง/มีทติ้ง/มีททิ่ง|term|การประชุม
Conference Call|คอนเฟอเรนซ์คอล/คอนเฟอร์เรนซ์คอล|term|
Video Conference|วิดีโอคอนเฟอเรนซ์/วิดีโอคอนเฟอร์เรนซ์|term|
Project|โปรเจกต์/โปรเจค/โปรเจ็ค/โปรเจ็กต์/โปรเจคท์/โปรเจ็คท์/โปรเจ็คต์|term|โครงการ
Project Manager|โปรเจกต์เมเนเจอร์/โปรเจคแมเนเจอร์/โปรเจคเมเนเจอร์|term|ผู้จัดการโครงการ
PM|พีเอ็ม|acronym|Project Manager
PMO|พีเอ็มโอ|acronym|Project Management Office
Stakeholder|สเตกโฮลเดอร์/สเตคโฮลเดอร์|term|ผู้มีส่วนได้ส่วนเสีย
Scope|สโคป|term|ขอบเขตงาน
Requirement|รีไควร์เมนต์/รีไควร์เม้นท์|term|ความต้องการ
Approval|แอพพรูฟวัล/อะพรูฟวัล|term|การอนุมัติ
Approve|แอพพรูฟ/อะพรูฟ/แอปพรูฟ|term|อนุมัติ
Sign-off|ไซน์ออฟ|term|ลงนามรับรอง
Priority|ไพรออริตี้/ไพรออริตี|term|ลำดับความสำคัญ
Status|สเตตัส|term|
Issue|อิชชู่/อิสชู|term|ประเด็น
Risk|ริสก์|term|ความเสี่ยง
Plan|แพลน|term|แผน
Planning|แพลนนิ่ง|term|การวางแผน
Schedule|สเกจูล/สเกดดูล/ชิดดูล|term|กำหนดการ
Process|โพรเซส/โปรเซส|term|กระบวนการ
Workflow|เวิร์กโฟลว์/เวิร์คโฟลว์|term|ขั้นตอนงาน
Procedure|โพรซีเยอร์|term|ขั้นตอน
Policy|โพลิซี/พอลิซี่/พอลิซี|term|นโยบาย
Guideline|ไกด์ไลน์|term|แนวทาง
Framework|เฟรมเวิร์ก/เฟรมเวิร์ค|term|
Template|เทมเพลต/เทมเพลท|term|แบบฟอร์ม
Checklist|เช็กลิสต์/เช็คลิสต์|term|
Best Practice|เบสต์แพรกทิส/เบสท์แพรคทิส|term|แนวปฏิบัติที่ดี
Benchmark|เบนช์มาร์ก/เบนช์มาร์ค|term|
Pilot|ไพลอต/ไพล็อต|term|โครงการนำร่อง
POC|พีโอซี|acronym|Proof of Concept
MVP|เอ็มวีพี|acronym|Minimum Viable Product
Phase|เฟส|term|ระยะ
Go-live|โกไลฟ์|term|เริ่มใช้งานจริง
Launch|ลอนช์/ล้อนช์|term|เปิดตัว
Rollout|โรลเอาต์/โรลเอาท์|term|
Implementation|อิมพลีเมนเทชั่น/อิมพลิเมนเทชั่น|term|การนำไปใช้
Deployment|ดีพลอยเมนต์/ดีพลอยเม้นท์|term|
Integration|อินทิเกรชั่น/อินทิเกรชัน|term|
Migration|ไมเกรชั่น/ไมเกรชัน|term|
Testing|เทสติ้ง|term|
UAT|ยูเอที|acronym|User Acceptance Test
SIT|เอสไอที|acronym|System Integration Test
Go/No-go|โกโนโก|term|
Resource|รีซอร์ส/รีซอส|term|ทรัพยากร
Manpower|แมนพาวเวอร์|term|กำลังคน
Headcount|เฮดเคานต์/เฮดเค้าท์|term|จำนวนพนักงาน
Workload|เวิร์กโหลด/เวิร์คโหลด|term|ภาระงาน
Capacity|คาปาซิตี้/แคปาซิตี้|term|
Bottleneck|บอตเทิลเน็ก/บอทเทิลเน็ค|term|คอขวด
Escalate|เอสคาเลต/เอสคาเลท|term|ส่งต่อให้ผู้บริหารพิจารณา
Alignment|อะไลน์เมนต์/อะไลน์เม้นท์|term|
Align|อะไลน์|term|
Sync|ซิงก์/ซิงค์|term|
Recap|รีแคป|term|สรุปทบทวน
Summary|ซัมมารี่/ซัมมารี|term|สรุป
Highlight|ไฮไลต์/ไฮไลท์|term|
Key Takeaway|คีย์เทคอะเวย์|term|
Next Step|เน็กซ์สเต็ป/เน็กซ์สเตป|term|ขั้นตอนถัดไป
Owner|โอนเนอร์|term|ผู้รับผิดชอบ
Assign|แอสไซน์/อะไซน์|term|มอบหมาย
Task|ทาสก์/ทาส์ก|term|งาน
Pending|เพนดิ้ง/เพ็นดิ้ง|term|ค้างอยู่
On track|ออนแทร็ก/ออนแทร็ค|term|เป็นไปตามแผน
Delay|ดีเลย์|term|ล่าช้า
Postpone|โพสต์โพน|term|เลื่อน
Cancel|แคนเซิล/แคนเซิ่ล|term|ยกเลิก
Confirm|คอนเฟิร์ม|term|ยืนยัน
Deliverable|ดีลิเวอร์เรเบิล/เดลิเวอเรเบิล|term|ผลงานที่ต้องส่งมอบ
Output|เอาต์พุต/เอาท์พุต|term|
Outcome|เอาต์คัม/เอาท์คัม|term|
Impact|อิมแพกต์/อิมแพค/อิมแพ็ค|term|ผลกระทบ
Solution|โซลูชั่น/โซลูชัน|term|
Option|ออปชั่น/อ๊อปชั่น/ออปชัน|term|ทางเลือก
Scenario|ซีนาริโอ|term|
Case Study|เคสสตัดี้/เคสสตัดดี้|term|
Insight|อินไซต์/อินไซท์|term|
Data|ดาต้า|term|ข้อมูล
Analytics|อนาไลติกส์/อะนาลิติกส์|term|
Analysis|อนาลิซิส/อะนาไลซิส|term|การวิเคราะห์
Revenue|เรเวนิว/เรฟเวนิว|term|รายได้
Profit|พรอฟิต/โพรฟิต|term|กำไร
Margin|มาร์จิ้น/มาร์จิน|term|อัตรากำไร
Gross Margin|กรอสมาร์จิ้น/กรอสมาร์จิน|term|อัตรากำไรขั้นต้น
Net Profit|เน็ตพรอฟิต|term|กำไรสุทธิ
Cost|คอสต์|term|ต้นทุน
Cost Saving|คอสต์เซฟวิ่ง|term|
Cash Flow|แคชโฟลว์|term|กระแสเงินสด
CAPEX|แคปเอ็กซ์/แคปเปกซ์|acronym|Capital Expenditure, รายจ่ายลงทุน
OPEX|โอเพ็กซ์/โอเปกซ์|acronym|Operating Expenditure, ค่าใช้จ่ายดำเนินงาน
EBITDA|อีบิทดา/อีบิดด้า/อีบิตด้า|acronym|กำไรก่อนดอกเบี้ย ภาษี ค่าเสื่อมราคาและค่าตัดจำหน่าย
P&L|พีแอนด์แอล|acronym|Profit and Loss, งบกำไรขาดทุน
ROI|อาร์โอไอ|acronym|Return on Investment
YoY|วายโอวาย|acronym|Year on Year, เทียบปีก่อน
QoQ|คิวโอคิว|acronym|Quarter on Quarter, เทียบไตรมาสก่อน
MoM|เอ็มโอเอ็ม|acronym|Month on Month
FY|เอฟวาย|acronym|Fiscal Year, ปีงบประมาณ
Q1|คิววัน|acronym|ไตรมาสที่ 1
Q2|คิวทู|acronym|ไตรมาสที่ 2
Q3|คิวทรี|acronym|ไตรมาสที่ 3
Q4|คิวโฟร์|acronym|ไตรมาสที่ 4
H1|เอชวัน|acronym|ครึ่งปีแรก
H2|เอชทู|acronym|ครึ่งปีหลัง
Sales|เซลส์/เซลล์ส|term|ฝ่ายขาย / ยอดขาย
Sales Target|เซลส์ทาร์เก็ต|term|เป้ายอดขาย
Pipeline|ไปป์ไลน์/ไปไลน์|term|
Lead|ลีด|term|ลูกค้าเป้าหมาย
Conversion Rate|คอนเวอร์ชั่นเรท/คอนเวอร์ชันเรต|term|
Marketing|มาร์เก็ตติ้ง/มาร์เกตติ้ง|term|การตลาด
Campaign|แคมเปญ/แคมเปน|term|
Promotion|โปรโมชั่น/โปรโมชัน|term|
Branding|แบรนดิ้ง|term|
Brand|แบรนด์|term|
Market Share|มาร์เก็ตแชร์|term|ส่วนแบ่งการตลาด
Segment|เซกเมนต์/เซกเม้นท์/เซ็กเมนต์|term|
Positioning|โพซิชันนิ่ง/โพสิชั่นนิ่ง|term|
Competitor|คอมเพทิเตอร์/คอมพีทิเตอร์|term|คู่แข่ง
Customer|คัสโตเมอร์/คัสตอมเมอร์|term|ลูกค้า
Client|ไคลเอนต์/ไคลเอ็นท์/ไคลเอ้นท์|term|ลูกค้า
Customer Experience|คัสโตเมอร์เอ็กซ์พีเรียนซ์|term|
Customer Journey|คัสโตเมอร์เจอร์นีย์|term|
Partner|พาร์ตเนอร์/พาร์ทเนอร์|term|คู่ค้า
Partnership|พาร์ตเนอร์ชิป/พาร์ทเนอร์ชิพ|term|
Vendor|เวนเดอร์/เว็นเดอร์|term|ผู้ขาย
Supplier|ซัพพลายเออร์|term|ผู้จัดหา
Outsource|เอาต์ซอร์ส/เอาท์ซอร์ส/เอาท์ซอส|term|จ้างภายนอก
Contract|คอนแทรกต์/คอนแทรค|term|สัญญา
Quotation|โควเทชั่น/โควเทชัน|term|ใบเสนอราคา
Invoice|อินวอยซ์|term|ใบแจ้งหนี้
Purchase Order|เพอร์เชสออร์เดอร์|term|ใบสั่งซื้อ
PO|พีโอ|acronym|Purchase Order
TOR|ทีโออาร์|acronym|Terms of Reference, ขอบเขตงาน
RFP|อาร์เอฟพี|acronym|Request for Proposal
RFQ|อาร์เอฟคิว|acronym|Request for Quotation
RFI|อาร์เอฟไอ|acronym|Request for Information
BOQ|บีโอคิว|acronym|Bill of Quantities
MOU|เอ็มโอยู|acronym|Memorandum of Understanding, บันทึกความเข้าใจ
NDA|เอ็นดีเอ|acronym|Non-Disclosure Agreement
SLA|เอสแอลเอ|acronym|Service Level Agreement
SOP|เอสโอพี|acronym|Standard Operating Procedure
Procurement|โพรเคียวร์เมนต์/โปรเคียวเม้นท์|term|การจัดซื้อจัดจ้าง
Supply Chain|ซัพพลายเชน|term|ห่วงโซ่อุปทาน
Logistics|โลจิสติกส์/โลจิสติก|term|
Inventory|อินเวนทอรี่/อินเวนทอรี|term|สินค้าคงคลัง
Stock|สต็อก/สต๊อก|term|
Warehouse|แวร์เฮาส์/แวร์เฮ้าส์|term|คลังสินค้า
Operation|โอเปอเรชั่น/โอเปอเรชัน|term|ปฏิบัติการ
Production|โปรดักชั่น/โปรดักชัน|term|การผลิต
Quality|ควอลิตี้/ควอลิตี|term|คุณภาพ
QC|คิวซี|acronym|Quality Control
QA|คิวเอ|acronym|Quality Assurance
Product|โปรดักต์/โปรดัก/โพรดักต์/โปรดักท์|term|ผลิตภัณฑ์
Service|เซอร์วิส|term|บริการ
Feature|ฟีเจอร์|term|
Prototype|โปรโตไทป์/โพรโทไทป์|term|ต้นแบบ
Design|ดีไซน์|term|การออกแบบ
Innovation|อินโนเวชั่น/อินโนเวชัน|term|นวัตกรรม
Digital Transformation|ดิจิทัลทรานส์ฟอร์เมชั่น/ดิจิตอลทรานส์ฟอร์เมชั่น|term|
Automation|ออโตเมชั่น/ออโตเมชัน|term|ระบบอัตโนมัติ
Platform|แพลตฟอร์ม|term|
System|ซิสเต็ม/ซิสเท็ม|term|ระบบ
Infrastructure|อินฟราสตรักเจอร์/อินฟราสตรัคเจอร์|term|โครงสร้างพื้นฐาน
Cloud|คลาวด์|term|
Server|เซิร์ฟเวอร์/เซิฟเวอร์|term|
Database|ดาต้าเบส/ดาตาเบส|term|ฐานข้อมูล
Cybersecurity|ไซเบอร์ซีเคียวริตี้/ไซเบอร์ซิเคียวริตี้|term|ความมั่นคงปลอดภัยไซเบอร์
Security|ซีเคียวริตี้/ซิเคียวริตี้|term|ความปลอดภัย
Big Data|บิ๊กดาต้า|term|
Machine Learning|แมชชีนเลิร์นนิง/แมชีนเลิร์นนิ่ง|term|
Generative AI|เจเนอเรทีฟเอไอ/เจนเนอเรทีฟเอไอ|term|
Chatbot|แชตบอต/แชทบอท|term|
Application|แอปพลิเคชัน/แอพพลิเคชั่น/แอปพลิเคชั่น|term|
Website|เว็บไซต์|term|
Mobile App|โมบายแอป/โมบายแอพ|term|
User|ยูสเซอร์/ยูเซอร์|term|ผู้ใช้งาน
Login|ล็อกอิน/ล็อคอิน|term|
Account|แอคเคานต์/แอ็กเคานต์/แอคเคาท์|term|บัญชี
Bug|บั๊ก|term|ข้อผิดพลาดของระบบ
Ticket|ทิกเก็ต/ทิกเกต|term|
Support|ซัพพอร์ต/ซัพพอร์ท|term|
Maintenance|เมนเทนแนนซ์/เมนเทแนนซ์|term|การบำรุงรักษา
Upgrade|อัปเกรด/อัพเกรด|term|
Version|เวอร์ชั่น/เวอร์ชัน|term|
Release|รีลีส|term|
Backup|แบ็กอัป/แบ็คอัพ/แบคอัพ|term|
Downtime|ดาวน์ไทม์|term|
Incident|อินซิเดนต์/อินซิเด้นท์|term|เหตุการณ์ผิดปกติ
BCP|บีซีพี|acronym|Business Continuity Plan
DR|ดีอาร์|acronym|Disaster Recovery
IT|ไอที|acronym|
AI|เอไอ|acronym|
API|เอพีไอ|acronym|
ERP|อีอาร์พี|acronym|
CRM|ซีอาร์เอ็ม|acronym|
HR|เอชอาร์|acronym|Human Resources, ฝ่ายทรัพยากรบุคคล
PR|พีอาร์|acronym|Public Relations, ประชาสัมพันธ์
CEO|ซีอีโอ|acronym|Chief Executive Officer
CFO|ซีเอฟโอ|acronym|Chief Financial Officer
COO|ซีโอโอ|acronym|Chief Operating Officer
CTO|ซีทีโอ|acronym|Chief Technology Officer
CIO|ซีไอโอ|acronym|Chief Information Officer
CMO|ซีเอ็มโอ|acronym|Chief Marketing Officer
CRO|ซีอาร์โอ|acronym|Chief Risk Officer
CHRO|ซีเอชอาร์โอ|acronym|Chief Human Resources Officer
CDO|ซีดีโอ|acronym|Chief Digital / Data Officer
MD|เอ็มดี|acronym|Managing Director, กรรมการผู้จัดการ
VP|วีพี|acronym|Vice President
SVP|เอสวีพี|acronym|Senior Vice President
EVP|อีวีพี|acronym|Executive Vice President
AVP|เอวีพี|acronym|Assistant Vice President
BOD|บีโอดี|acronym|Board of Directors, คณะกรรมการบริษัท
Board|บอร์ด|term|คณะกรรมการ
Chairman|แชร์แมน/แชร์แมนส์|term|ประธาน
Director|ไดเรกเตอร์/ไดเร็กเตอร์|term|กรรมการ / ผู้อำนวยการ
Manager|เมเนเจอร์/แมเนเจอร์/เมเนจเจอร์|term|ผู้จัดการ
Management|แมเนจเมนต์/เมเนจเม้นท์/แมเนจเม้นท์|term|ฝ่ายบริหาร
Executive|เอ็กเซ็กคิวทีฟ/เอ็กเซคคิวทีฟ|term|ผู้บริหาร
Supervisor|ซูเปอร์ไวเซอร์/ซุปเปอร์ไวเซอร์|term|หัวหน้างาน
Team Lead|ทีมลีด|term|หัวหน้าทีม
Leader|ลีดเดอร์|term|ผู้นำ
Leadership|ลีดเดอร์ชิป/ลีดเดอร์ชิพ|term|ภาวะผู้นำ
Committee|คอมมิตตี/คอมมิตตี้|term|คณะกรรมการ
Secretary|เซเครทารี/เซคเครทารี่|term|เลขานุการ
Moderator|มอเดอเรเตอร์/โมเดอเรเตอร์|term|ผู้ดำเนินรายการ
Facilitator|ฟาซิลิเทเตอร์/ฟาซิลิเตเตอร์|term|ผู้อำนวยความสะดวกการประชุม
Speaker|สปีกเกอร์/สปีคเกอร์|term|ผู้บรรยาย
Department|ดีพาร์ตเมนต์/ดีพาร์ทเม้นท์|term|ฝ่าย / แผนก
Organization|ออร์แกไนเซชั่น/ออแกไนเซชั่น|term|องค์กร
Corporate|คอร์ปอเรต/คอร์ปอเรท|term|องค์กร
Headquarters|เฮดควอเตอร์/เฮดควอร์เตอร์|term|สำนักงานใหญ่
Branch|บรานช์|term|สาขา
Subsidiary|ซับซิเดียรี/ซับซิเดียรี่|term|บริษัทย่อย
Holding|โฮลดิ้ง|term|
Shareholder|แชร์โฮลเดอร์|term|ผู้ถือหุ้น
Investor|อินเวสเตอร์/อินเวสเตอร์ส|term|นักลงทุน
Investor Relations|อินเวสเตอร์รีเลชั่นส์/อินเวสเตอร์รีเลชั่น|term|นักลงทุนสัมพันธ์
IR|ไออาร์|acronym|Investor Relations
Investment|อินเวสต์เมนต์/อินเวสเม้นท์|term|การลงทุน
Dividend|ดิวิเดนด์/ดีวิเดนด์|term|เงินปันผล
Valuation|แวลูเอชั่น/แวลูเอชัน|term|การประเมินมูลค่า
IPO|ไอพีโอ|acronym|Initial Public Offering
M&A|เอ็มแอนด์เอ|acronym|Mergers and Acquisitions, การควบรวมกิจการ
Merger|เมอร์เจอร์|term|การควบรวม
Acquisition|แอคควิซิชั่น/อะควิซิชั่น|term|การเข้าซื้อกิจการ
JV|เจวี|acronym|Joint Venture, กิจการร่วมค้า
Joint Venture|จอยต์เวนเจอร์/จอยท์เวนเจอร์|term|กิจการร่วมค้า
Startup|สตาร์ตอัป/สตาร์ทอัพ|term|
SME|เอสเอ็มอี|acronym|
B2B|บีทูบี|acronym|
B2C|บีทูซี|acronym|
ESG|อีเอสจี|acronym|Environmental, Social and Governance
CSR|ซีเอสอาร์|acronym|Corporate Social Responsibility
Sustainability|ซัสเทนะบิลิตี้/ซัสเทนเนบิลิตี้|term|ความยั่งยืน
Net Zero|เน็ตซีโร่/เน็ตซีโร|term|
Carbon Neutral|คาร์บอนนิวทรัล|term|
Governance|กัฟเวอร์แนนซ์/กัฟเวอแนนซ์|term|ธรรมาภิบาล
Corporate Governance|คอร์ปอเรตกัฟเวอร์แนนซ์|term|การกำกับดูแลกิจการ
CG|ซีจี|acronym|Corporate Governance
GRC|จีอาร์ซี|acronym|Governance, Risk and Compliance
Compliance|คอมพลายแอนซ์/คอมไพลแอนซ์|term|การปฏิบัติตามกฎเกณฑ์
Audit|ออดิต/ออดิท|term|การตรวจสอบ
Auditor|ออดิเตอร์/ออดิทเตอร์|term|ผู้สอบบัญชี / ผู้ตรวจสอบ
Internal Audit|อินเทอร์นอลออดิต/อินเทอร์นอลออดิท|term|ตรวจสอบภายใน
Risk Management|ริสก์แมเนจเมนต์/ริสก์เมเนจเม้นท์|term|การบริหารความเสี่ยง
PDPA|พีดีพีเอ|acronym|พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล
ISO|ไอเอสโอ|acronym|
SWOT|สวอต/สว็อต|acronym|Strengths, Weaknesses, Opportunities, Threats
RACI|เรซี่/ราซี่|acronym|Responsible, Accountable, Consulted, Informed
NPS|เอ็นพีเอส|acronym|Net Promoter Score
CX|ซีเอ็กซ์|acronym|Customer Experience
UX|ยูเอ็กซ์|acronym|User Experience
UI|ยูไอ|acronym|User Interface
AOB|เอโอบี|acronym|Any Other Business, เรื่องอื่นๆ
FYI|เอฟวายไอ|acronym|For Your Information
ASAP|เอเอสเอพี|acronym|As Soon As Possible
ETA|อีทีเอ|acronym|Estimated Time of Arrival
Training|เทรนนิ่ง/เทรนนิง|term|การฝึกอบรม
Onboarding|ออนบอร์ดดิ้ง/ออนบอร์ดิ้ง|term|การรับพนักงานใหม่
Coaching|โค้ชชิ่ง/โค้ชชิง|term|
Mentor|เมนเทอร์/เมนเทอ|term|พี่เลี้ยง
Recruitment|รีครูตเมนต์/รีครูทเม้นท์|term|การสรรหา
Interview|อินเทอร์วิว/อินเตอร์วิว|term|สัมภาษณ์
Performance|เพอร์ฟอร์แมนซ์/เพอร์ฟอร์มานซ์|term|ผลการปฏิบัติงาน
Performance Review|เพอร์ฟอร์แมนซ์รีวิว|term|การประเมินผลงาน
Appraisal|แอพเพรซัล/อะเพรซัล|term|การประเมิน
Bonus|โบนัส|term|
Incentive|อินเซนทีฟ/อินเซนทิฟ|term|ค่าตอบแทนจูงใจ
Commission|คอมมิชชั่น/คอมมิชชัน|term|ค่านายหน้า
Salary|แซลารี่/ซาลารี่|term|เงินเดือน
Benefit|เบเนฟิต/เบเนฟิท|term|สวัสดิการ / ประโยชน์
Overtime|โอเวอร์ไทม์|term|ทำงานล่วงเวลา
OT|โอที|acronym|Overtime
Work from Home|เวิร์กฟรอมโฮม/เวิร์คฟรอมโฮม|term|ทำงานที่บ้าน
WFH|ดับเบิลยูเอฟเอช|acronym|Work from Home
Hybrid Work|ไฮบริดเวิร์ก/ไฮบริดเวิร์ค|term|
Culture|คัลเจอร์|term|วัฒนธรรมองค์กร
Engagement|เอนเกจเมนต์/เอ็นเกจเม้นท์|term|ความผูกพัน
Mindset|มายด์เซ็ต/มายด์เซต|term|กรอบความคิด
Skill|สกิล|term|ทักษะ
Upskill|อัปสกิล/อัพสกิล|term|
Reskill|รีสกิล|term|
Talent|ทาเลนต์/ทาเลนท์|term|
Career Path|แคเรียร์พาธ|term|เส้นทางอาชีพ
Succession Plan|ซัคเซสชั่นแพลน|term|แผนสืบทอดตำแหน่ง
Restructure|รีสตรักเจอร์/รีสตรัคเจอร์|term|ปรับโครงสร้าง
Transformation|ทรานส์ฟอร์เมชั่น/ทรานส์ฟอร์เมชัน|term|
Change Management|เชนจ์แมเนจเมนต์/เชนจ์เมเนจเม้นท์|term|การบริหารการเปลี่ยนแปลง
Agile|อไจล์/เอไจล์|term|
Scrum|สกรัม|term|
Sprint|สปรินต์/สปริ้นท์|term|
Backlog|แบ็กล็อก/แบ็คล็อก|term|
Retrospective|เรโทรสเปกทีฟ/เรโทรสเปคทีฟ|term|
Stand-up|สแตนด์อัป/สแตนด์อัพ|term|ประชุมสั้นประจำวัน
One-on-one|วันออนวัน|term|ประชุมตัวต่อตัว
Town Hall|ทาวน์ฮอลล์/ทาวน์ฮอล|term|การประชุมพนักงานทั้งองค์กร
Offsite|ออฟไซต์/ออฟไซท์|term|การประชุมนอกสถานที่
Seminar||term|
Webinar|เว็บบินาร์/เว็บินาร์|term|
Event|อีเวนต์/อีเว้นท์/อีเว้นต์|term|
Venue|เวนิว/เวนู|term|สถานที่จัดงาน
Slide|สไลด์|term|
Handout|แฮนด์เอาต์/แฮนด์เอาท์|term|เอกสารประกอบ
Attachment|แอตแทชเมนต์/แอทแทชเม้นท์|term|ไฟล์แนบ
Document|ด็อกคิวเมนต์/ดอคคิวเม้นท์|term|เอกสาร
File|ไฟล์|term|
Folder|โฟลเดอร์|term|
Link|ลิงก์/ลิงค์|term|
Email|อีเมล/อีเมล์|term|
Calendar|แคลเลนดาร์/คาเลนดาร์|term|ปฏิทิน
Invitation|อินวิเทชั่น/อินวิเตชั่น|term|หนังสือเชิญ
Reminder|รีไมนด์เดอร์/รีมายเดอร์|term|การแจ้งเตือน
Zoom|ซูม|term|
Microsoft Teams|ไมโครซอฟต์ทีมส์/ไมโครซอฟท์ทีมส์|term|
Google Meet|กูเกิลมีต/กูเกิ้ลมีท|term|
Webex|เว็บเอ็กซ์|term|
Mute|มิวต์/มิวท์|term|ปิดเสียง
Share Screen|แชร์สกรีน|term|แชร์หน้าจอ
Record|รีคอร์ด/เรคคอร์ด|term|บันทึก
Recording|รีคอร์ดดิ้ง/เรคคอร์ดดิ้ง|term|
Transcript|ทรานสคริปต์/ทรานสคริปท์|term|บทถอดเสียง
Consensus|คอนเซนซัส|term|ฉันทามติ
Quorum|โควรัม|term|องค์ประชุม
Voting|โวตติ้ง/โหวตติ้ง|term|การลงคะแนน
Vote|โหวต/โหวด|term|ลงคะแนน
Proxy|พร็อกซี/พร็อกซี่|term|ผู้รับมอบฉันทะ
Resolution|เรโซลูชั่น/เรโซลูชัน|term|มติ
Motion|โมชั่น/โมชัน|term|ญัตติ
Objection|ออบเจกชั่น/อ็อบเจ็กชั่น|term|การคัดค้าน
Record Date|เรคคอร์ดเดต/รีคอร์ดเดท|term|วันกำหนดรายชื่อผู้ถือหุ้น
Annual Report|แอนนวลรีพอร์ต/แอนนวลรีพอร์ท|term|รายงานประจำปี
Financial Statement|ไฟแนนเชียลสเตตเมนต์/ไฟแนนเชียลสเตทเม้นท์|term|งบการเงิน
Balance Sheet|บาลานซ์ชีต/บาลานซ์ชีท|term|งบดุล / งบแสดงฐานะการเงิน
Audit Committee|ออดิตคอมมิตตี/ออดิทคอมมิตตี้|term|คณะกรรมการตรวจสอบ
Nomination Committee|โนมิเนชั่นคอมมิตตี้|term|คณะกรรมการสรรหา
Remuneration|เรมูเนอเรชั่น/รีมูเนอเรชั่น|term|ค่าตอบแทน
Independent Director|อินดิเพนเดนต์ไดเรกเตอร์/อินดีเพนเดนท์ไดเร็กเตอร์|term|กรรมการอิสระ
Insider|อินไซเดอร์|term|ผู้มีข้อมูลภายใน
Conflict of Interest|คอนฟลิกต์ออฟอินเทอเรสต์/คอนฟลิคออฟอินเทอเรส|term|ความขัดแย้งทางผลประโยชน์
Related Party|รีเลทเท็ดปาร์ตี้/รีเลทเต็ดปาร์ตี้|term|บุคคลที่เกี่ยวโยงกัน
Disclosure|ดิสโคลเชอร์/ดิสโคลชเชอร์|term|การเปิดเผยข้อมูล
Whistleblowing|วิสเซิลโบลว์วิ่ง/วิสเซิลโบลวิ่ง|term|การแจ้งเบาะแส
Anti-corruption|แอนตี้คอร์รัปชั่น/แอนตี้คอรัปชั่น|term|ต่อต้านคอร์รัปชัน
SET|เอสอีที|acronym|ตลาดหลักทรัพย์แห่งประเทศไทย
mai|เอ็มเอไอ|acronym|ตลาดหลักทรัพย์ เอ็ม เอ ไอ
GDP|จีดีพี|acronym|
Inflation|อินเฟลชั่น/อินเฟรชั่น|term|เงินเฟ้อ
Recession|รีเซสชั่น/รีเซสชัน|term|เศรษฐกิจถดถอย
Exchange Rate|เอ็กซ์เชนจ์เรต/เอ็กซ์เชนจ์เรท|term|อัตราแลกเปลี่ยน
Growth|โกรท|term|การเติบโต
Trend|เทรนด์|term|แนวโน้ม
Opportunity|ออพพอร์ทูนิตี้/อ็อปพอร์ทูนิตี้|term|โอกาส
Challenge|ชาเลนจ์/ชาเล้นจ์|term|ความท้าทาย
Gap||term|ช่องว่าง
Root Cause|รูทคอส/รูตคอส|term|สาเหตุที่แท้จริง
Lesson Learned|เลสซันเลิร์น/เลสซั่นเลิร์นด์|term|บทเรียน
Improvement|อิมพรูฟเมนต์/อิมพรูฟเม้นท์|term|การปรับปรุง
Continuous Improvement|คอนตินิวอัสอิมพรูฟเมนต์|term|การปรับปรุงอย่างต่อเนื่อง
Lean||term|
Six Sigma|ซิกซ์ซิกมา|term|
Kaizen|ไคเซ็น/ไคเซน|term|
PDCA|พีดีซีเอ|acronym|Plan-Do-Check-Act
Efficiency|เอฟฟิเชียนซี/อิฟฟิเชียนซี่|term|ประสิทธิภาพ
Productivity|โปรดักทิวิตี้/โปรดัคทิวิตี้|term|ผลิตภาพ
Optimize|ออปติไมซ์/ออพติไมซ์|term|ปรับให้เหมาะสม
Scale up|สเกลอัป/สเกลอัพ|term|ขยายขนาด
Expansion|เอ็กซ์แพนชั่น/เอ็กซ์แพนชัน|term|การขยายกิจการ
Go-to-market|โกทูมาร์เก็ต|term|
Business Model|บิสซิเนสโมเดล/บิสสิเนสโมเดล|term|โมเดลธุรกิจ
Business Plan|บิสซิเนสแพลน/บิสสิเนสแพลน|term|แผนธุรกิจ
Business Unit|บิสซิเนสยูนิต/บิสสิเนสยูนิท|term|หน่วยธุรกิจ
BU||acronym|Business Unit
Feasibility Study|ฟีซิบิลิตี้สตัดี้/ฟีสิบิลิตี้สตัดดี้|term|การศึกษาความเป็นไปได้
Due Diligence|ดิวดิลิเจนซ์/ดิวดิลิเจ้นซ์|term|การตรวจสอบสถานะ
Negotiation|เนโกชิเอชั่น/เนโกทิเอชั่น|term|การเจรจา
Deal||term|ข้อตกลง
Win-win|วินวิน|term|
Commitment|คอมมิตเมนต์/คอมมิทเม้นท์|term|ความมุ่งมั่น
Accountability|แอคเคาท์ทะบิลิตี้/อะคาวน์ทะบิลิตี้|term|ความรับผิดชอบ
Ownership|โอนเนอร์ชิป/โอนเนอร์ชิพ|term|ความเป็นเจ้าของ
Empower|เอ็มพาวเวอร์/เอมพาวเวอร์|term|
Collaboration|คอลแลบอเรชั่น/คอลลาบอเรชั่น|term|การทำงานร่วมกัน
Communication|คอมมิวนิเคชั่น/คอมมิวนิเคชัน|term|การสื่อสาร
Transparency|ทรานสแพเรนซี/ทรานสแพเรนซี่|term|ความโปร่งใส
Integrity|อินทิกริตี้/อินเทกริตี้|term|ความซื่อตรง
Vision|วิชั่น/วิชัน|term|วิสัยทัศน์
Mission|มิชชั่น/มิชชัน|term|พันธกิจ
Core Value|คอร์แวลู/คอร์แวลูว์|term|ค่านิยมหลัก
Objective|ออบเจกทีฟ/อ็อบเจ็คทีฟ|term|วัตถุประสงค์
Goal||term|เป้าหมาย
"""

# Thai meeting vocabulary: correct written forms (no repair rules).
THAI = """
วาระการประชุม|agenda
ระเบียบวาระ|agenda item
เรื่องแจ้งเพื่อทราบ|matters for information
เรื่องเพื่อพิจารณา|matters for consideration
เรื่องอื่นๆ|any other business
รับรองรายงานการประชุม|adopt the minutes
รายงานการประชุม|minutes
มติที่ประชุม|resolution
ที่ประชุมมีมติ|the meeting resolved
อนุมัติ|approve
ไม่อนุมัติ|reject
เห็นชอบ|endorse
รับทราบ|acknowledge
มอบหมาย|assign
ผู้รับผิดชอบ|person responsible
กำหนดเสร็จ|due date
ติดตามความคืบหน้า|follow up progress
ความคืบหน้า|progress
องค์ประชุม|quorum
ครบองค์ประชุม|quorum present
ประธานที่ประชุม|chair of the meeting
ประธานกรรมการ|chairman of the board
รองประธาน|vice chair
เลขานุการคณะกรรมการ|board secretary
เลขานุการบริษัท|company secretary
กรรมการผู้จัดการ|managing director
กรรมการบริษัท|director
กรรมการอิสระ|independent director
กรรมการตรวจสอบ|audit committee member
คณะกรรมการบริหาร|executive committee
คณะกรรมการบริหารความเสี่ยง|risk management committee
คณะกรรมการสรรหาและพิจารณาค่าตอบแทน|nomination and remuneration committee
คณะกรรมการกำกับดูแลกิจการ|corporate governance committee
คณะทำงาน|working group
อนุกรรมการ|subcommittee
ผู้ถือหุ้น|shareholder
ผู้รับมอบฉันทะ|proxy holder
หนังสือมอบฉันทะ|proxy form
มอบฉันทะ|appoint a proxy
ลงทะเบียน|register
ลงคะแนนเสียง|vote
ออกเสียงลงคะแนน|cast votes
บัตรลงคะแนน|ballot
เห็นด้วย|approve (vote)
ไม่เห็นด้วย|disapprove (vote)
งดออกเสียง|abstain
บัตรเสีย|void ballot
เสียงข้างมาก|majority
มติเป็นเอกฉันท์|unanimous resolution
คะแนนเสียงไม่น้อยกว่าสามในสี่|not less than three quarters of the votes
ผู้ตรวจสอบการนับคะแนน|vote inspector
ผู้สอบบัญชี|auditor
ค่าสอบบัญชี|audit fee
งบการเงิน|financial statements
งบแสดงฐานะการเงิน|statement of financial position
งบกำไรขาดทุนเบ็ดเสร็จ|statement of comprehensive income
งบกระแสเงินสด|cash flow statement
รายงานประจำปี|annual report
แบบ 56-1 One Report|Form 56-1 One Report
เงินปันผล|dividend
จ่ายเงินปันผล|pay dividend
งดจ่ายเงินปันผล|omit dividend
เงินปันผลระหว่างกาล|interim dividend
ทุนสำรองตามกฎหมาย|legal reserve
กำไรสะสม|retained earnings
ทุนจดทะเบียน|registered capital
ทุนชำระแล้ว|paid-up capital
เพิ่มทุน|capital increase
ลดทุน|capital reduction
หุ้นสามัญ|ordinary share
หุ้นบุริมสิทธิ|preferred share
มูลค่าที่ตราไว้|par value
ใบสำคัญแสดงสิทธิ|warrant
หุ้นกู้|debenture
ค่าตอบแทนกรรมการ|directors' remuneration
บำเหน็จกรรมการ|directors' bonus
กรรมการที่ออกตามวาระ|directors retiring by rotation
แต่งตั้งกรรมการ|appoint directors
แก้ไขเพิ่มเติมข้อบังคับ|amend the articles of association
หนังสือบริคณห์สนธิ|memorandum of association
ข้อบังคับบริษัท|articles of association
นิติบุคคล|juristic person
บริษัทมหาชนจำกัด|public company limited
บริษัทจดทะเบียน|listed company
บริษัทย่อย|subsidiary
บริษัทร่วม|associated company
บริษัทในเครือ|affiliate
รายการที่เกี่ยวโยงกัน|connected transaction
รายการได้มาหรือจำหน่ายไป|acquisition or disposal transaction
ส่วนได้เสีย|interest (stake)
ผู้มีส่วนได้เสีย|interested party
การเปิดเผยสารสนเทศ|information disclosure
ข้อมูลภายใน|inside information
การกำกับดูแลกิจการที่ดี|good corporate governance
จรรยาบรรณธุรกิจ|code of conduct
ต่อต้านการทุจริตคอร์รัปชัน|anti-corruption
การควบคุมภายใน|internal control
การตรวจสอบภายใน|internal audit
การบริหารความเสี่ยง|risk management
ความเสี่ยงด้านปฏิบัติการ|operational risk
ความเสี่ยงด้านเครดิต|credit risk
ความเสี่ยงด้านสภาพคล่อง|liquidity risk
แผนธุรกิจ|business plan
แผนกลยุทธ์|strategic plan
แผนปฏิบัติการ|action plan
แผนงบประมาณ|budget plan
งบประมาณประจำปี|annual budget
งบลงทุน|capital budget
ค่าใช้จ่ายในการดำเนินงาน|operating expenses
รายได้จากการขาย|sales revenue
รายได้รวม|total revenue
กำไรขั้นต้น|gross profit
กำไรสุทธิ|net profit
ขาดทุนสุทธิ|net loss
อัตรากำไรสุทธิ|net profit margin
ยอดขาย|sales
เป้ายอดขาย|sales target
ส่วนแบ่งการตลาด|market share
ไตรมาส|quarter
ครึ่งปีแรก|first half
ปีงบประมาณ|fiscal year
เทียบกับช่วงเดียวกันของปีก่อน|compared with the same period last year
เพิ่มขึ้นร้อยละ|increased by percent
ลดลงร้อยละ|decreased by percent
ตัวชี้วัด|indicator (KPI)
ผลการดำเนินงาน|operating results
ผลประกอบการ|performance (results)
แนวโน้ม|trend
สภาวะเศรษฐกิจ|economic conditions
อัตราแลกเปลี่ยน|exchange rate
อัตราดอกเบี้ย|interest rate
ต้นทุนทางการเงิน|finance cost
สภาพคล่อง|liquidity
กระแสเงินสด|cash flow
หนี้สิน|liabilities
สินทรัพย์|assets
ส่วนของผู้ถือหุ้น|shareholders' equity
อัตราส่วนหนี้สินต่อทุน|debt to equity ratio
การลงทุน|investment
การร่วมลงทุน|joint investment
การควบรวมกิจการ|merger
การเข้าซื้อกิจการ|acquisition
ปรับโครงสร้างองค์กร|organisational restructuring
โครงสร้างองค์กร|organisation structure
การจัดซื้อจัดจ้าง|procurement
ประกวดราคา|tender
คู่ค้า|business partner
ผู้รับเหมา|contractor
สัญญาจ้าง|service contract
ข้อกำหนดและเงื่อนไข|terms and conditions
ระยะเวลาดำเนินการ|duration
ส่งมอบงาน|deliver the work
ตรวจรับงาน|accept the work
งวดงาน|instalment of work
ค่าปรับ|penalty
บันทึกข้อตกลง|memorandum of understanding
บันทึกข้อความ|internal memo
หนังสือเวียน|circular letter
หนังสือเชิญประชุม|meeting invitation
เอกสารประกอบการประชุม|meeting documents
ระเบียบปฏิบัติ|regulation (internal)
นโยบายบริษัท|company policy
ประกาศบริษัท|company announcement
คำสั่งแต่งตั้ง|appointment order
ทรัพยากรบุคคล|human resources
อัตรากำลัง|manpower
การประเมินผลการปฏิบัติงาน|performance appraisal
การพัฒนาบุคลากร|staff development
สวัสดิการพนักงาน|employee benefits
ความปลอดภัยอาชีวอนามัย|occupational health and safety
ความรับผิดชอบต่อสังคม|social responsibility
การพัฒนาอย่างยั่งยืน|sustainable development
ก๊าซเรือนกระจก|greenhouse gas
ลูกค้าเป้าหมาย|target customer
กลุ่มเป้าหมาย|target group
ช่องทางการจัดจำหน่าย|distribution channel
ความพึงพอใจของลูกค้า|customer satisfaction
ข้อร้องเรียน|complaint
คุณภาพการให้บริการ|service quality
ระบบสารสนเทศ|information system
ความมั่นคงปลอดภัยของข้อมูล|information security
ข้อมูลส่วนบุคคล|personal data
การคุ้มครองข้อมูลส่วนบุคคล|personal data protection
การประชุมผ่านสื่ออิเล็กทรอนิกส์|electronic meeting
พระราชกำหนดว่าด้วยการประชุมผ่านสื่ออิเล็กทรอนิกส์|Emergency Decree on Electronic Meetings
ข้อเสนอแนะ|recommendation
ข้อสังเกต|observation
ข้อซักถาม|question
ชี้แจง|clarify
อภิปราย|discuss
พิจารณา|consider
ทบทวน|review
เสนอ|propose
ผู้เสนอ|proposer
ผู้รับรอง|seconder
ปิดประชุม|close the meeting
เปิดประชุม|open the meeting
เลื่อนการประชุม|postpone the meeting
ประชุมครั้งถัดไป|next meeting
นัดหมาย|appointment
ผู้เข้าร่วมประชุม|attendees
ผู้ไม่มาประชุม|absentees
ผู้ลาประชุม|apologies
ผู้เข้าร่วมสังเกตการณ์|observers
"""


# Longer words that contain a pack spelling but are other words (คีย์บอร์ด holds บอร์ด, โปรไฟล์ holds ไฟล์ ...): the rules leave
# them as they are. Found by matching every spelling of every pack against a 62k-word Thai dictionary + app/loanwords.tsv.
GUARD = """
อินเตอร์ลิงก์ อินเทอร์ลิงก์ อินเตอร์ลิงค์ อินเทอร์ลิงค์
กราฟิกดีไซน์ กรู๊ฟไรเดอร์ กลิงค์ คีบอร์ด คีย์บอร์ด ค้างสต๊อก ซันไมโครซิสเต็มส์ ซีบอร์ด ฐานข้อมูลดาต้า ดีบั๊ก
ดีวิชั่น ดีไซน์เนอร์ ติดบอร์ด ตู้เอทีเอ็ม ทรูวิชั่น ทีดีอาร์ไอ ทีโอที นปุงสกลิงค์ น้ำสต๊อก บรัสเซลส์ ปุงลิงค์
ปุลลิงค์ พร้อมมิตรโปรดักชั่น พาร์ทเนอร์ส ฟอสเตอร์แอนด์พาร์ตเนอร์ มิชชันนารี รีแบรนด์ รุสต๊อก ลิปซิงค์ ลีดส์
ล้างสต๊อก วิชั่นส์ วีพีเอ็น สตรีลิงค์ สปอร์ตไรเดอร์ สเกตบอร์ด สเปกโทรสโคป หัสดีลิงค์ อลิงค์ ออดิทอเรียม
อากริคัลเจอร์ อิตถีลิงค์ อินพุตเอาต์พุต อินเทรนด์ อิมแพ็คอารีน่า อีสเทิร์นซีบอร์ด อีเมลล์ อเวนิว
เครื่องฉายสไลด์ เงินโบนัส เชียร์ลีดดิ้ง เชียร์ลีดเดอร์ เซลส์สลิป เซลส์แมน เทเลวิชั่น เฟสติวัล เฟสบุ๊ก เฟสบุ๊ค
เมนบอร์ด เว็บบอร์ด เอสจีโอที เอไอที เอไอเอส เอไอเอสพลัส แอดมิชชั่น แอร์พอร์ตลิงก์ แอลซีดี โปรดักส์ โปรเจคเตอร์
โปรเซสเซอร์ โปรไฟล์ โพรเซสเซอร์ โพรไฟล์ โหวตเสียง ไทยทิกเก็ตเมเจอร์ ไวท์บอร์ด ไออาร์พีซี ไอโอที
""".split()


def rows():
    out = []
    for line in TABLE.strip().splitlines():
        term, variants, cat, note = (line.split("|") + ["", "", ""])[:4]
        out.append((term.strip(), [v.strip() for v in variants.split("/") if v.strip()], cat.strip() or "term", note.strip()))
    for line in THAI.strip().splitlines():
        term, note = (line.split("|") + [""])[:2]
        out.append((term.strip(), [], "term", note.strip()))
    return out
