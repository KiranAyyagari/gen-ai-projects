### Customer Support Smart Triage System

Building upon the foundational workflow, this advanced system introduces more granular classification and a multi-step triage process for technical support issues.

### Workflow Diagram

<img src="diagrams/img2.png" alt="Smart Triage System Workflow Diagram" style="max-width: 100%; height: auto;" />

### Workflow Steps

1.  **Input:** A customer message.
2.  **Classify Intent:** The first agent classifies the inquiry's purpose into one of three categories: **Sales Inquiry**, **Technical Support**, or **Billing Question**.
3.  **Initial Routing:**
    *   If **Sales Inquiry** ➞ **Sales Agent** (process ends).
    *   If **Billing Question** ➞ **Billing Agent** (process ends).
    *   If **Technical Support** ➞ **Triage Urgency**.
4.  **Triage Urgency (Technical Support only):** A second agent analyzes the message to determine its urgency level: **High** or **Standard**.
5.  **Secondary Routing (Urgency-based):**
    *   If **High Urgency** ➞ **Escalation Agent**.
    *   If **Standard Urgency** ➞ **Automated Support Agent**.
6.  **Generate Response:**
    *   **Sales Agent:** Provides links to pricing or sales contacts.
    *   **Billing Agent:** Directs users to their account dashboard.
    *   **Escalation Agent:** Creates a high-priority support ticket and notifies the user that a human agent will assist shortly.
    *   **Automated Support Agent:** Offers standard solutions or links to FAQs (e.g., "Have you tried restarting? Here’s a link to our documentation.").

---

## Key Benefits

*   **Increased Efficiency:** Automates the manual sorting of customer inquiries, freeing up team members for more complex tasks.
*   **Improved Customer Experience:** Provides faster, more consistent, and more accurate responses, leading to higher satisfaction.
*   **Scalability:** Creates a flexible and modular framework that can be easily expanded with new categories, agents, and tools over time.
*   **24/7 Availability:** Offers immediate, automated responses and routing even outside of standard business hours.

## Proposed Technology Stack

*   **Orchestration:** LangGraph for building the stateful, multi-agent workflows.
*   **Language Models:** Large Language Models (LLMs) for classification, analysis, and response generation.
*   **Data Validation:** Pydantic for ensuring structured and consistent data exchange between workflow components.
