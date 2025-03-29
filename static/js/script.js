document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const toolButtons = document.querySelectorAll('.tool-button');

    function addMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        const messageParagraph = document.createElement('p');
        messageParagraph.innerHTML = message.replace(/\n/g, '<br>');
        const messageTime = document.createElement('div');
        messageTime.classList.add('message-time');
        const now = new Date();
        messageTime.textContent = `${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
        messageContent.appendChild(messageParagraph);
        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(messageTime);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('message', 'bot-message', 'typing-indicator');
        typingDiv.innerHTML = '<div class="message-content"><p>Sharma is thinking...</p></div>';
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return typingDiv;
    }

    async function sendMessage(message) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(`Server error: ${data.error || 'Unknown'}`);
            console.log('Server response:', data);  // Debug log
            return data.response;
        } catch (error) {
            console.error('Fetch error:', error);
            return `Sorry, I couldn’t process that: ${error.message}`;
        }
    }

    async function handleSendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        userInput.value = '';
        addMessage(message, 'user');
        const typingIndicator = showTypingIndicator();
        const response = await sendMessage(message);
        chatMessages.removeChild(typingIndicator);
        addMessage(response, 'Sharma');
    }

    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSendMessage(); });
    toolButtons.forEach(button => button.addEventListener('click', (e) => {
        userInput.value = e.target.getAttribute('data-tool') === 'pregnancy' ? 'Can you tell me about pregnancy ' : '';
        userInput.focus();
    }));

    async function loadChatHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            chatMessages.innerHTML = '';
            if (data.history && data.history.length > 0) {
                data.history.forEach(item => addMessage(item.message, item.sender));
            } else {
                addMessage("Hello! I'm Sharma, your maternal and child health assistant...", 'Sharma');
            }
        } catch (error) {
            console.error('Error loading history:', error);
            addMessage("Oops, couldn’t load chat history!", 'Sharma');
        }
    }
    
    loadChatHistory();
    userInput.focus();
});