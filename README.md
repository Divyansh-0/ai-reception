Here's the entire content converted into Markdown:

```markdown
# AI Medical Receptionist

## Overview
This project is an AI-powered medical receptionist designed to assist users by determining the urgency of their medical concerns, providing immediate advice, and forwarding messages to the doctor if necessary. The AI can handle conversations, assess emergencies, and offer clarifications based on prior advice given.

## Features
- **Emergency Detection:** The AI assesses user input to determine if the situation is an emergency.
- **Location-Based Response:** Provides estimated arrival times and specific instructions based on the user's location.
- **Contextual Response Generation:** Queries a pre-loaded medical document to provide contextually accurate advice.
- **Follow-Up Clarifications:** Handles follow-up questions or clarifications based on the previous advice given.


## Setup

### Prerequisites
- Python 3.10 or later
- Streamlit for the UI
- Chroma for vector storage
- Google Generative AI for embeddings and chat functionality

### Installation
Clone the repository:

```bash
git clone https://github.com/Divyansh-0/ai-reception.git
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Usage
Once the application is running, you can start interacting with the AI Medical Receptionist. Enter your query in the input field, and the AI will guide you through determining if it's an emergency, providing appropriate advice, or forwarding a message to the doctor.

## Structure
- **app.py:** The main application script.
- **requirements.txt:** Lists all Python dependencies.
- **Medical_Receptionist_QnA.pdf:** A PDF document loaded for contextual queries.
```

