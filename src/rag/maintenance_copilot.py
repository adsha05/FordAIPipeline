from src.rag.retriever import retrieve_docs


def generate_answer(question: str):
    docs = retrieve_docs(question, top_k=2)

    context = "\n\n".join(
        [f"Source: {doc['source']}\n{doc['text']}" for doc in docs]
    )

    answer = f"""
Maintenance Copilot Answer

Question:
{question}

Likely Issue:
Based on the retrieved maintenance knowledge, this issue may be related to the systems described in the source documents.

Recommended Actions:
"""

    q = question.lower()

    recommendations = []

    if "overheat" in q or "engine temp" in q or "temperature" in q:
        recommendations.append("Inspect coolant level, radiator condition, thermostat, water pump, and engine temperature sensor.")

    if "oil" in q or "pressure" in q:
        recommendations.append("Check oil level, oil filter, oil pump pressure, and oil pressure sensor calibration.")

    if "battery" in q or "voltage" in q:
        recommendations.append("Test battery health, alternator output, terminal corrosion, and charging history.")

    if "brake" in q:
        recommendations.append("Inspect brake pads, rotors, brake fluid, and recent braking behavior.")

    if "transmission" in q:
        recommendations.append("Check transmission fluid, diagnostic codes, gear sensor, and prior service history.")

    if not recommendations:
        recommendations.append("Schedule a standard preventive maintenance inspection and review recent telematics/service history.")

    for i, rec in enumerate(recommendations, start=1):
        answer += f"\n{i}. {rec}"

    answer += "\n\nRetrieved Sources:\n"

    for doc in docs:
        answer += f"- {doc['source']}\n"

    answer += "\nRetrieved Context:\n"
    answer += context

    return {
        "question": question,
        "answer": answer,
        "sources": [doc["source"] for doc in docs]
    }


if __name__ == "__main__":
    result = generate_answer(
        "Vehicle has high engine temperature and low oil pressure. What should we inspect?"
    )

    print(result["answer"])