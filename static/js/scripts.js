document.addEventListener('DOMContentLoaded', () => {
    const newBotBtn = document.getElementById('new-bot-btn');
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const newBotModal = document.getElementById('new-bot-modal');
    const createBotBtn = document.getElementById('create-bot-btn');
    const cancelCreateBtn = document.getElementById('cancel-create-btn');
    const botInfoModal = document.getElementById('bot-info-modal');
    const botPersona = document.getElementById('bot-persona');
    const botEndpoint = document.getElementById('bot-endpoint');
    const closeInfoBtn = document.getElementById('close-info-btn');
    const confirmDeleteModal = document.getElementById('confirm-delete-modal');
    const confirmDeleteBtn = document.getElementById('confirm-delete-btn');
    const cancelDeleteBtn = document.getElementById('cancel-delete-btn');
    const botsList = document.getElementById('bots-list');
    const messages = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingSpinner = document.createElement('div');
    const configureEndpointsBtn = document.getElementById('configure-endpoints-btn');
    const configureEndpointsModal = document.getElementById('configure-endpoints-modal');
    const endpointsList = document.getElementById('endpoints-list');
    const addEndpointBtn = document.getElementById('add-endpoint-btn');
    const closeEndpointConfigBtn = document.getElementById('close-endpoint-config-btn');

    loadingSpinner.classList.add('spinner');
    loadingSpinner.style.display = 'none';
    sendButton.parentNode.insertBefore(loadingSpinner, sendButton.nextSibling);

    let selectedBot = null;
    let botToDelete = null;

    async function fetchBots() {
        try {
            const response = await fetch('/bots');
            const bots = await response.json();
            botsList.innerHTML = '';
            bots.forEach(bot => {
                const botItem = document.createElement('div');
                botItem.classList.add('bot-item');
                botItem.dataset.botName = bot.name;
                botItem.innerHTML = `
                    <span>${bot.name}</span>
                    <div class="bot-buttons">
                        <button class="info-btn" data-bot-name="${bot.name}">i</button>
                        <button class="delete-btn" data-bot-name="${bot.name}">x</button>
                    </div>
                `;
                botsList.appendChild(botItem);
            });
            populateEndpointsSelect();
        } catch (error) {
            console.error('Error fetching bots:', error);
        }
    }

    async function fetchChatHistory(botName) {
        try {
            const response = await fetch(`/chat_history?bot_name=${botName}`);
            const history = await response.json();
            messages.innerHTML = '';
            history.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            history.forEach(entry => {
                const messageDiv = document.createElement('div');
                messageDiv.classList.add('message');
                const sanitizedMessage = DOMPurify.sanitize(marked.parse(entry.conversation)); // Ensure using the correct function
                if (entry.role === 'user') {
                    messageDiv.classList.add('user-message');
                    messageDiv.innerHTML = `<strong>You</strong><br><span class="timestamp">${new Date(entry.timestamp).toLocaleString()}</span><br><p>${sanitizedMessage}</p>`;
                } else {
                    messageDiv.classList.add('bot-message');
                    messageDiv.innerHTML = `<strong>${botName}</strong><br><span class="timestamp">${new Date(entry.timestamp).toLocaleString()}</span><br><p>${sanitizedMessage}</p>`;
                }
                messages.appendChild(messageDiv);
            });
            messages.scrollTop = messages.scrollHeight;
        } catch (error) {
            console.error('Error fetching chat history:', error);
        }
    }

    newBotBtn.addEventListener('click', () => {
        newBotModal.style.display = 'flex';
    });


    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            if (document.body.classList.contains('dark-mode')) {
                darkModeToggle.innerHTML = '<i class="fa-solid fa-sun"></i>';
            } else {
                darkModeToggle.innerHTML = '<i class="fa-solid fa-moon"></i>';
            }
        });
    }

    createBotBtn.addEventListener('click', async () => {
        const botName = document.getElementById('new-bot-name').value;
        const botPersona = document.getElementById('new-bot-persona').value;
        const endpointName = document.getElementById('new-bot-endpoint').value;

        if (!botName || !botPersona || !endpointName) {
            alert('Please fill out all required fields');
            return;
        }

        try {
            const response = await fetch('/create_bot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name: botName, persona: botPersona, endpoint_name: endpointName })
            });
            if (response.ok) {
                fetchBots();
                newBotModal.style.display = 'none';
                document.getElementById('new-bot-name').value = '';
                document.getElementById('new-bot-persona').value = '';
            } else {
                alert('Error creating bot');
            }
        } catch (error) {
            console.error('Error creating bot:', error);
        }
    });

    cancelCreateBtn.addEventListener('click', () => {
        newBotModal.style.display = 'none';
    });

    botsList.addEventListener('click', async (e) => {
        if (e.target.classList.contains('info-btn')) {
            const botName = e.target.getAttribute('data-bot-name');
            try {
                await populateEndpointsSelect(); // Ensure endpoints are fetched before setting the value
                const response = await fetch(`/bot_info?bot_name=${botName}`);
                const data = await response.json();
                botPersona.textContent = data.persona;
                botEndpoint.value = data.endpoint_name;
                console.log('Bot Info:', data); // Debugging statement
                console.log('Bot Endpoint Value Set To:', botEndpoint.value); // Debugging statement
                botInfoModal.style.display = 'flex';
            } catch (error) {
                console.error('Error fetching bot info:', error);
            }
        } else if (e.target.classList.contains('delete-btn')) {
            botToDelete = e.target.getAttribute('data-bot-name');
            confirmDeleteModal.style.display = 'flex';
        } else {
            const botItem = e.target.closest('.bot-item');
            if (botItem) {
                selectedBot = botItem.dataset.botName;
                document.querySelectorAll('.bot-item').forEach(item => item.classList.remove('selected'));
                botItem.classList.add('selected');
                await fetchChatHistory(selectedBot);
            }
        }
    });

    confirmDeleteBtn.addEventListener('click', async () => {
        try {
            await fetch(`/delete_bot?bot_name=${botToDelete}`, {
                method: 'DELETE'
            });
            fetchBots();
            confirmDeleteModal.style.display = 'none';
            botToDelete = null;
        } catch (error) {
            console.error('Error deleting bot:', error);
        }
    });

    cancelDeleteBtn.addEventListener('click', () => {
        confirmDeleteModal.style.display = 'none';
    });

    closeInfoBtn.addEventListener('click', () => {
        botInfoModal.style.display = 'none';
    });

    botEndpoint.addEventListener('change', async () => {
        const newEndpoint = botEndpoint.value;
        try {
            const response = await fetch('/update_bot_api_endpoint', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name: selectedBot, endpoint_name: newEndpoint })
            });
            if (!response.ok) {
                alert('Error updating endpoint');
            }
        } catch (error) {
            console.error('Error updating endpoint:', error);
        }
    });

    async function sendMessage() {
        const userMessage = userInput.value;
        if (!userMessage.trim()) return; // Don't send empty messages

        const userMessageDiv = document.createElement('div');
        userMessageDiv.classList.add('message', 'user-message');
        const sanitizedUserMessage = DOMPurify.sanitize(marked.parse(userMessage)); // Ensure using the correct function
        userMessageDiv.innerHTML = `<strong>You</strong><br><span class="timestamp">${new Date().toLocaleString()}</span><br><p>${sanitizedUserMessage}</p>`;
        messages.appendChild(userMessageDiv);
        messages.scrollTop = messages.scrollHeight;

        userInput.disabled = true;
        sendButton.disabled = true;
        loadingSpinner.style.display = 'inline-block';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ user_input: userMessage, bot_name: selectedBot })
            });
            const data = await response.json();
            const botMessageDiv = document.createElement('div');
            botMessageDiv.classList.add('message', 'bot-message');
            const sanitizedBotMessage = DOMPurify.sanitize(marked.parse(data.response)); // Ensure using the correct function
            botMessageDiv.innerHTML = `<strong>${selectedBot}</strong><br><span class="timestamp">${new Date().toLocaleString()}</span><br><p>${sanitizedBotMessage}</p>`;
            messages.appendChild(botMessageDiv);
            messages.scrollTop = messages.scrollHeight;
        } catch (error) {
            console.error('Error sending message:', error);
        } finally {
            userInput.value = '';
            userInput.disabled = false;
            sendButton.disabled = false;
            loadingSpinner.style.display = 'none';
        }
    }

    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    async function fetchEndpoints() {
        try {
            const response = await fetch('/api_endpoints');
            const endpoints = await response.json();
            endpointsList.innerHTML = '';
            endpoints.forEach(endpoint => {
                const endpointItem = document.createElement('div');
                endpointItem.classList.add('endpoint-item');
                endpointItem.innerHTML = `
                    <span>${endpoint.name}</span>
                    <div class="endpoint-buttons">
                        <button class="delete-endpoint-btn" data-endpoint-name="${endpoint.name}">x</button>
                    </div>
                `;
                endpointsList.appendChild(endpointItem);
            });
            configureEndpointsModal.style.display = 'flex';
        } catch (error) {
            console.error('Error fetching endpoints:', error);
        }
    }

    configureEndpointsBtn.addEventListener('click', fetchEndpoints);

    addEndpointBtn.addEventListener('click', async () => {
        const endpointName = document.getElementById('new-endpoint-name').value;
        const endpointUrl = document.getElementById('new-endpoint-url').value;
        const endpointModel = document.getElementById('new-endpoint-model').value;
        const endpointToken = document.getElementById('new-endpoint-token').value;
        const contextLength = document.getElementById('new-endpoint-context-length').value;
        const reservedTokens = document.getElementById('new-endpoint-reserved-tokens').value;

        if (!endpointName || !endpointUrl || !endpointModel || !contextLength || !reservedTokens) {
            alert('Please fill out all required fields');
            return;
        }

        try {
            const response = await fetch('/add_api_endpoint', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: endpointName,
                    api_url: endpointUrl,
                    model: endpointModel,
                    token: endpointToken,
                    context_length: parseInt(contextLength, 10),
                    reserved_tokens_for_response: parseInt(reservedTokens, 10)
                })
            });
            if (response.ok) {
                document.getElementById('new-endpoint-name').value = '';
                document.getElementById('new-endpoint-url').value = '';
                document.getElementById('new-endpoint-model').value = '';
                document.getElementById('new-endpoint-token').value = '';
                document.getElementById('new-endpoint-context-length').value = '';
                document.getElementById('new-endpoint-reserved-tokens').value = '';
                fetchEndpoints();
            } else {
                alert('Error adding endpoint');
            }
        } catch (error) {
            console.error('Error adding endpoint:', error);
        }
    });

    closeEndpointConfigBtn.addEventListener('click', () => {
        configureEndpointsModal.style.display = 'none';
    });

    endpointsList.addEventListener('click', async (e) => {
        if (e.target.classList.contains('delete-endpoint-btn')) {
            const endpointName = e.target.getAttribute('data-endpoint-name');
            try {
                const response = await fetch(`/delete_api_endpoint?name=${endpointName}`, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    fetchEndpoints();
                } else {
                    alert('Error deleting endpoint');
                }
            } catch (error) {
                console.error('Error deleting endpoint:', error);
            }
        }
    });

    async function populateEndpointsSelect() {
        try {
            const response = await fetch('/api_endpoints');
            const endpoints = await response.json();
            const endpointSelectNewBot = document.getElementById('new-bot-endpoint');
            const endpointSelectBotInfo = document.getElementById('bot-endpoint');
            endpointSelectNewBot.innerHTML = '';
            endpointSelectBotInfo.innerHTML = '';
            endpoints.forEach(endpoint => {
                const optionNewBot = document.createElement('option');
                const optionBotInfo = document.createElement('option');
                optionNewBot.value = endpoint.name;
                optionNewBot.textContent = endpoint.name;
                optionBotInfo.value = endpoint.name;
                optionBotInfo.textContent = endpoint.name;
                endpointSelectNewBot.appendChild(optionNewBot);
                endpointSelectBotInfo.appendChild(optionBotInfo);
            });
            console.log('Endpoints populated:', endpoints); // Debugging statement
        } catch (error) {
            console.error('Error fetching endpoints:', error);
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        const menuToggle = document.getElementById('menu-toggle');
        const container = document.getElementById('container');
    
        menuToggle.addEventListener('click', function() {
            container.classList.toggle('sidebar-visible');
        });
    });
    fetchBots();
});
