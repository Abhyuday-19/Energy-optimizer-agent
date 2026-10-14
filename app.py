import json
from datetime import datetime
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Energy Optimizer Agent", page_icon="⚡", layout="centered")

PUNJAB_SLABS = [
    (100, 3.85),
    (300, 5.25),
    (500, 6.95),
    (float("inf"), 8.15),
]
FIXED_CHARGE = 130
ELECTRICITY_DUTY = 0.05
FREE_UNITS = 300


def calculate_slab_bill(units):
    if units <= 0:
        return 0.0
    remaining, prev_limit, total = units, 0, 0.0
    for limit, rate in PUNJAB_SLABS:
        slab_units = min(remaining, limit - prev_limit)
        if slab_units <= 0:
            break
        total += slab_units * rate
        remaining -= slab_units
        prev_limit = limit
        if remaining <= 0:
            break
    return total


def calculate_bill(units):
    if units <= FREE_UNITS:
        return {
            "units": units, "free_scheme_applied": True,
            "energy_charge": 0.0, "fixed_charge": 0.0, "duty": 0.0, "total": 0.0,
        }
    energy_charge = calculate_slab_bill(units)
    duty = energy_charge * ELECTRICITY_DUTY
    total = energy_charge + FIXED_CHARGE + duty
    return {
        "units": units, "free_scheme_applied": False,
        "energy_charge": round(energy_charge, 2), "fixed_charge": FIXED_CHARGE,
        "duty": round(duty, 2), "total": round(total, 2),
    }


def units_to_next_slab_saving(units):
    if units <= FREE_UNITS:
        return None
    over = units - FREE_UNITS
    return {
        "units_over_free_limit": round(over, 1),
        "advice": (f"Aap free {FREE_UNITS}-unit limit se {over:.1f} units zyada use kar rahe ho. "
                   f"Agar is mahine {over:.1f} units kam kar lein, poora bill Rs 0 ho sakta hai "
                   f"(sirf extra units nahi, poora bill — Punjab ke cliff rule ki wajah se)."),
    }


@st.cache_resource(show_spinner="Loading model and knowledge base...")
def load_resources():
    with open("data/appliance_wattage.json") as f:
        appliance_wattage = json.load(f)
    with open("data/energy_tips.json") as f:
        energy_tips = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    tip_texts = [t["tip"] for t in energy_tips]
    tip_embeddings = model.encode(tip_texts, normalize_embeddings=True).astype("float32")

    dimension = tip_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(tip_embeddings)

    return appliance_wattage, energy_tips, model, index


appliance_wattage, energy_tips, model, index = load_resources()


def retrieve_tips(query, k=5):
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q_vec, k)
    return [{**energy_tips[i], "relevance": round(float(s), 3)} for s, i in zip(scores[0], idx[0])]


def get_current_season():
    month = datetime.now().month
    if month in (3, 4, 5, 6):
        return "Seasonal-Summer"
    elif month in (11, 12, 1, 2):
        return "Seasonal-Winter"
    elif month == 10:
        return "Seasonal-Festival"
    else:
        return None


def merge_seasonal_tips(base_tips, all_tips, season, extra=2):
    if season is None:
        return base_tips
    existing_ids = {t["id"] for t in base_tips}
    seasonal_pool = [t for t in all_tips if t["category"] == season and t["id"] not in existing_ids]
    boosted = []
    for t in seasonal_pool[:extra]:
        t = dict(t)
        t["relevance"] = "seasonal-boost"
        boosted.append(t)
    return boosted + base_tips


def energy_agent(usage_hours, k_tips=5):
    consumption = []
    for appliance, hours in usage_hours.items():
        watt = appliance_wattage.get(appliance, 0)
        monthly_kwh = (watt * hours * 30) / 1000
        consumption.append({"appliance": appliance, "hours_per_day": hours, "monthly_kwh": round(monthly_kwh, 1)})
    consumption.sort(key=lambda x: x["monthly_kwh"], reverse=True)

    total_units = round(sum(c["monthly_kwh"] for c in consumption), 1)
    bill = calculate_bill(total_units)
    proximity_advice = units_to_next_slab_saving(total_units)

    top_appliances = [c["appliance"] for c in consumption[:3]]
    query = "Energy saving tips for: " + ", ".join(top_appliances) if top_appliances else "general energy saving"
    tips = retrieve_tips(query, k=k_tips)

    season = get_current_season()
    tips = merge_seasonal_tips(tips, energy_tips, season)

    return {
        "consumption_breakdown": consumption,
        "total_monthly_units": total_units,
        "bill": bill,
        "proximity_advice": proximity_advice,
        "recommended_tips": tips,
        "current_season": season,
    }


st.title("⚡ Energy Consumption Optimizer Agent")
st.caption("India-specific (Punjab/PSPCL) · RAG-powered tips · Built for Build Next 2026")

st.markdown(
    "Apne ghar ke appliances aur unka daily usage daalo — ye AI agent tumhara "
    "estimated bill nikalega aur personalized energy-saving tips dega."
)

st.subheader("Apna usage daalo")

selected_appliances = st.multiselect(
    "Appliances select karo jo tum roz use karte ho",
    options=list(appliance_wattage.keys()),
    default=["Air Conditioner (1.5 Ton, 3 Star)", "Refrigerator (Double Door, 250L)"],
)

usage_hours = {}
if selected_appliances:
    st.write("Har appliance ke liye roz kitne ghante use hota hai:")
    cols = st.columns(2)
    for i, appliance in enumerate(selected_appliances):
        with cols[i % 2]:
            hours = st.number_input(
                appliance, min_value=0.0, max_value=24.0, value=4.0, step=0.5,
                key=f"hrs_{appliance}",
            )
            usage_hours[appliance] = hours

if st.button("🔍 Check my bill & get tips", type="primary", disabled=not selected_appliances):
    result = energy_agent(usage_hours)

    st.subheader("📊 Consumption Breakdown")
    for c in result["consumption_breakdown"]:
        st.write(f"**{c['appliance']}** — {c['hours_per_day']} hrs/day → {c['monthly_kwh']} kWh/month")

    st.metric("Total Estimated Monthly Units", f"{result['total_monthly_units']} kWh")

    st.subheader("💰 Estimated Bill (PSPCL, Punjab)")
    bill = result["bill"]
    if bill["free_scheme_applied"]:
        st.success(f"Bill: ₹0 — within the {FREE_UNITS} free units/month scheme! 🎉")
    else:
        st.error(f"Total Bill: ₹{bill['total']}")
        st.write(f"Energy charge: ₹{bill['energy_charge']} | Fixed charge: ₹{bill['fixed_charge']} | Duty: ₹{bill['duty']}")

    if result["proximity_advice"]:
        st.warning(result["proximity_advice"]["advice"])

    st.subheader("💡 Personalized Energy-Saving Tips")
    if result["current_season"]:
        st.caption(f"🗓️ Season-aware: {result['current_season'].replace('Seasonal-', '')} tips boosted for this time of year")
    for tip in result["recommended_tips"]:
        badge = "🌟 Seasonal" if tip["relevance"] == "seasonal-boost" else f"relevance: {tip['relevance']}"
        st.info(f"**[{tip['category']}]** {tip['tip']}  \n*({badge})*")

st.markdown("---")
st.caption(
    "⚠️ Billing rates are approximate (PSERC FY 2026-27 order) and for demonstration purposes. "
    "Verify against your actual PSPCL bill for exact figures."
)
