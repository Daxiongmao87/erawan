# 🐘Erawan

## Introduction
Erawan is an easy-to-use chatbot application designed to provide interactive and contextually aware conversations. It supports creating and managing multiple bots, each with a unique persona, and leverages advanced memory and keyword-based retrieval systems to enhance user interactions.

The name "Erawan" is inspired by the mythological white elephant known for its wisdom and memory, reflecting the application's capabilities.

## Important Features

- **Interactive Chat Interface**: Users can interact with various bots by sending messages and receiving responses based on the bots' unique personas.
- **Bot Personalization**: Each bot has a customizable persona that shapes its responses and behavior.
- **Contextual Conversations**: Bots remember previous interactions, allowing for coherent and contextually relevant conversations.
- **Keyword-Based Memory Retrieval**: Enhances bots' responses by retrieving relevant memories based on extracted keywords.
- **Conversation History**: Users can view past chats with a specific bot, enabling them to continue conversations seamlessly.
- **API Endpoint Management**: Supports dynamic updates to the API endpoints used by the bots, ensuring up-to-date and effective response generation.
- **Inner Monologue Feature**: Allows bots to internally analyze and reflect on conversations before responding, leading to more thoughtful replies.
- **Responsive Design**: Web-based interface accessible from any device with an internet connection.
- **Error Handling and Reliability**: Includes mechanisms to handle errors and retry API calls, ensuring a stable user experience.

## Roadmap Features

- **Memory Toggle**: Enable or disable memory functions for specific interactions.
- **Memory Reset**: Clear stored memories for a fresh start.
- **Internal Reasoning**: Enhance bots with the ability to perform internal reasoning for more insightful responses.
- **Multi-Agent Capabilities**: Support interactions involving multiple bots working together.
- **Document Imports**: Allow users to import documents for bots to read and reference in conversations.
- **Stable Diffusion API**: Integrate Stable Diffusion API for generating images based on bot responses and user interactions.

## Getting Started

To get started with 🐘Erawan, follow these steps:

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/erawan.git
   cd erawan
   ```

2. **Install Dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Set Up Configuration**:

- Create a config.ini file based on the provided config.ini.example template.
- Update the configuration settings as needed.

4. **Initialize the Database**:

   ```sh
   flask db upgrade
   ```
5. **Run the Application**:

   ```sh
   flask run
   ```
