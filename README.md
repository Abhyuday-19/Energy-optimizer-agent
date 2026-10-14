# ⚡ Energy Consumption Optimizer Agent

An AI agent that analyzes household appliance usage, estimates the electricity bill using
India's (Punjab/PSPCL) real tariff structure, and gives personalized, season-aware
energy-saving recommendations using Retrieval-Augmented Generation (RAG).

Built for **Build Next 2026** — the Kompact AI Hackathon by ZIROH Labs.

---

## 🎯 Problem Statement

Most people check their electricity bill but have no idea which appliance is actually
driving up the cost, or how close they are to crossing into a costlier billing slab.
Generic energy-saving advice online isn't personalized, and doesn't account for
India-specific billing rules like Punjab's unique 300-unit free scheme.

## 💡 What This Project Does

1. User selects the appliances they use and how many hours per day
2. The agent calculates **monthly consumption (kWh)** per appliance
3. It computes the **estimated electricity bill** using Punjab's (PSPCL) actual slab tariff
4. It retrieves the **most relevant energy-saving tips** for the user's specific high-consumption
   appliances using semantic search (RAG)
5. It **boosts seasonally relevant tips** automatically (summer AC tips in summer, Diwali
   lighting tips in festival season, etc.)

---

## 🌟 What Makes This Different

Most "energy optimizer" projects use generic/US-style flat billing and no localization.
This one is built specifically around real Indian constraints:

| Feature | Why it matters |
|---|---|
| **India-specific appliance & billing data** | 26 real Indian household appliances (not generic placeholders) |
| **Punjab (PSPCL) slab-wise billing, including the free-300-units "cliff" rule** | Punjab gives the first 300 units/month free — but cross that by even 1 unit and the *entire* bill is charged at slab rates, not just the excess. The agent explicitly warns users when they're close to this cliff, which is a real, high-impact, and rarely-modeled behavior. |
| **Season-aware tip boosting** | The agent detects the current month and automatically surfaces summer/winter/festival-specific tips, instead of generic advice year-round |
| **Lightweight, cost-effective architecture** | Uses a small 80MB embedding model (`all-MiniLM-L6-v2`) and rule-based billing logic (not an LLM call) — fast, cheap, and realistic for edge/low-resource deployment, in line with the hackathon's "cost-effective, efficient, edge-deployable" theme |

---

## 🏗️ Architecture
