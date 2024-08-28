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

    // Check and apply dark mode preference
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode === 'enabled') {
        document.body.classList.add('dark-mode');
        darkModeToggle.innerHTML = '<i class="fas fa-adjust"></i>';
    } else {
        document.body.classList.remove('dark-mode');
        darkModeToggle.innerHTML = '<i class="fas fa-adjust fa-flip-horizontal"></i>';
    }

async function fetchBots() {
    try {
        const response = await fetch('/bots');
        const bots = await response.json();
	console.log(bots);
        botsList.innerHTML = '';
        bots.forEach((bot, index) => {
            const botName = bot.name;
            const botItem = document.createElement('div');
            botItem.classList.add('bot-item');
            botItem.dataset.botName = botName;
            botItem.innerHTML = `
                <span>${botName}</span>
                <div class="bot-buttons">
                    <button class="info-btn small-button" data-bot-name="${botName}"><i class="fas fa-info-circle"></i></i></button>
                    <button class="delete-btn small-button" data-bot-name="${botName}"><i class="fa-solid fa-trash"></i></button>
                </div>
            `;
            botsList.appendChild(botItem);

            // Automatically select the first bot
            if (index === 0) {
                botItem.classList.add('selected');
                selectedBot = botName;
                fetchChatHistory(selectedBot);
            }
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


    // Existing event listener for dark mode toggle
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            if (document.body.classList.contains('dark-mode')) {
                darkModeToggle.innerHTML = '<i class="fas fa-adjust"></i>';
                localStorage.setItem('darkMode', 'enabled'); // Save preference
            } else {
                darkModeToggle.innerHTML = '<i class="fas fa-adjust fa-flip-horizontal"></i>';
                localStorage.setItem('darkMode', 'disabled'); // Save preference
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
                const endpointName = endpoint.configurations['name'];  // Extract the 'name' from the configurations
                const endpointItem = document.createElement('div');
                endpointItem.classList.add('endpoint-item');
                endpointItem.innerHTML = `
                    <span>${endpointName}</span>
                    <div class="endpoint-buttons">
                        <button class="delete-endpoint-btn small-button" data-endpoint-name="${endpointName}"><i class="fa-solid fa-trash"></i></button>
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
        const endpointName = document.getElementById('new-endpoint-name').value.trim();
        const endpointUrl = document.getElementById('new-endpoint-url').value.trim();
        const configText = document.getElementById('new-endpoint-config').value.trim();
    
        if (!endpointName || !endpointUrl) {
            alert('Please fill out all required fields');
            return;
        }
    
        // Parse the config input
        const configLines = configText.split('\n');
        const configEntries = [];
        for (let line of configLines) {
            const [variable_name, variable_value] = line.split('=').map(s => s.trim());
            if (variable_name && variable_value) {
                configEntries.push({ variable_name, variable_value });
            } else if (line.trim() !== '') {
                alert(`Invalid line: "${line}". Make sure it's in the format 'variable = value'`);
                return;
            }
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
                    config: configEntries
                })
            });
            if (response.ok) {
                document.getElementById('new-endpoint-name').value = '';
                document.getElementById('new-endpoint-url').value = '';
                document.getElementById('new-endpoint-config').value = '';
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
                const endpointName = endpoint.configurations['name'];  // Extract the 'name' from the configurations
                const optionNewBot = document.createElement('option');
                const optionBotInfo = document.createElement('option');
                optionNewBot.value = endpointName;
                optionNewBot.textContent = endpointName;
                optionBotInfo.value = endpointName;
                optionBotInfo.textContent = endpointName;
                endpointSelectNewBot.appendChild(optionNewBot);
                endpointSelectBotInfo.appendChild(optionBotInfo);
            });
            console.log('Endpoints populated:', endpoints); // Debugging statement
        } catch (error) {
            console.error('Error fetching endpoints:', error);
        }
    }

    const menuToggle = document.getElementById('menu-toggle');
    const container = document.getElementById('container');

    menuToggle.addEventListener('click', function() {
        console.log('Menu toggle clicked');
        container.classList.toggle('sidebar-visible');
        console.log(container.classList);  // Log to verify class is toggled
    });

    fetchBots();
});
