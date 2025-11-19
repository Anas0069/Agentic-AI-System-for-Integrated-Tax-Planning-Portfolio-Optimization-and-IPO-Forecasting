# update_data.py
import json, os, datetime
from llm_client import LLMClient

def update_dataset():
    llm = LLMClient()
    system_prompt = (
        "You are a tax-data extractor. Provide the latest Indian income tax slabs and GST rates for the current fiscal year. "
        "Output a JSON with keys: 'tax_slabs', 'deductions', 'gst_rates', 'notes'. Use only reliable sources; if uncertain, mark unchanged."
    )
    user_prompt = "Give me the current slabs, main deduction limits (80C, 80D), and common GST rates."
    print("Fetching latest tax data via LLM...")
    try:
        data_text = llm.ask(system_prompt, user_prompt)
        # attempt to parse JSON from LLM output
        raw = data_text[data_text.find("{"): data_text.rfind("}")+1]
        new_data = json.loads(raw)
    except Exception as e:
        print("Error parsing LLM result:", e)
        return False

    os.makedirs("data", exist_ok=True)
    if "tax_slabs" in new_data:
        with open("data/tax_slabs.json", "w") as f:
            json.dump(new_data["tax_slabs"], f, indent=2)
    if "deductions" in new_data:
        with open("data/deductions.json", "w") as f:
            json.dump(new_data["deductions"], f, indent=2)
    if "gst_rates" in new_data:
        with open("data/gst_rates.json", "w") as f:
            json.dump(new_data["gst_rates"], f, indent=2)

    version_info = {
        "last_updated": datetime.date.today().isoformat(),
        "budget_year": new_data.get("budget_year", "unknown"),
        "source": "LLM aggregated sources",
        "notes": new_data.get("notes", "")
    }
    with open("data/version_info.json", "w") as f:
        json.dump(version_info, f, indent=2)

    print("Dataset updated. Version info saved.")
    return True

if __name__ == "__main__":
    update_dataset()
