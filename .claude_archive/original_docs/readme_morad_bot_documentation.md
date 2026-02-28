# MoradBot — Documentation Folder

This folder contains the full structured documentation set for MoradBot.

These documents collectively define the business, market, product, system, and non-functional foundations of the project.

This is not code documentation.
This is decision documentation.

---

# 📂 Folder Structure

## 1️⃣ MRD — Market Requirements Document
**File:** MoradBot_Market_Requirements_Document_MRD_v1

Defines:
- Ideal Customer Profile (ICP)
- Core pain points
- Existing alternatives
- Market risks
- Whether the problem is real or optional

Purpose:
To validate that the market problem is worth solving.

---

## 2️⃣ BRD — Business Requirements Document
**File:** MoradBot_Business_Requirements_Document_BRD_v1

Defines:
- Core business objective
- Revenue model
- Pricing logic
- KPIs (3 & 6 months)
- Market constraints
- Go / No-Go decision

Purpose:
To define how the product makes money and why it should exist.

---

## 3️⃣ PRD — Product Requirements Document
**File:** MoradBot_Product_Requirements_Document_PRD_v1

Defines:
- Product description from user perspective
- Personas
- Must-have features
- Deferred features
- Out-of-scope boundaries
- User stories
- Clear MVP definition

Purpose:
To define what will be built in the first version.

---

## 4️⃣ NFR — Non-Functional Requirements
**File:** MoradBot_Non_Functional_Requirements_NFR_v1

Defines measurable system constraints:
- Performance targets
- Security requirements
- Scalability limits
- Reliability & recovery metrics
- Privacy & compliance constraints

Purpose:
To define system quality standards and operational expectations.

---

## 5️⃣ SRD — System Requirements Document
**File:** MoradBot_System_Requirements_Document_SRD_v1

Defines:
- System architecture overview
- Main components
- Data flows
- Technical requirements per component
- Constraints & assumptions
- Future integration paths

Purpose:
To translate product requirements into system-level structure.

---

## 6️⃣ Full Project Documentation
**File:** MoradBot_Full_Project_Documentation_v1

High-level master summary of:
- Scope
- Strategic decisions
- Boundaries
- Core operational model

Purpose:
Executive-level overview.

---

## 7️⃣ Extended Architecture Decisions
**File:** MoradBot_Extended_Architecture_Document_v1

Contains detailed operational decisions such as:
- Chat injection behavior
- Sync logic
- Monitoring
- Backup policy
- Incident response
- Secret management

Purpose:
Operational clarity and enforcement of architectural decisions.

---

# 📌 How to Read This Folder

If you are:

### 👔 Business / Strategy
Start with:
1. MRD
2. BRD

### 🎯 Product
Start with:
1. PRD
2. SRD

### 🛡 Security / Reliability
Start with:
1. NFR
2. SRD

### 👨‍💻 Engineering
Read in order:
1. PRD
2. NFR
3. SRD

---

# 🔒 Decision Integrity Rule

All documents represent explicit decisions.
No feature may be implemented that contradicts:
- PRD scope
- NFR constraints
- SRD architecture

Changes require formal revision and version increment.

---

# 🚀 Current Phase

Phase: MVP (FAQ Automation Only)

Not included:
- Multi-agent orchestration
- WhatsApp integration
- Order tracking
- Advanced analytics

Scope discipline is mandatory.

---

# 🧭 Versioning

Current documentation version: v1

Future updates must:
- Increment version
- Document change summary
- Preserve historical decisions

---

End of README
