// LinkedIn Post Generator JavaScript

// Configuration
const CONFIG = {
    API_ENDPOINT: '/generate',
    COPY_SUCCESS_DURATION: 2000,
    EXAMPLES: [
        'Artificial Intelligence',
        'Remote Work',
        'Digital Marketing',
        'Data Science',
        'Leadership',
        'Entrepreneurship',
        'Career Growth',
        'Technology Trends',
        'Work-Life Balance',
        'Innovation',
        'Cybersecurity',
        'Cloud Computing',
        'Machine Learning',
        'Blockchain',
        'Sustainability'
    ]
};

// DOM Elements
const elements = {
    form: null,
    topicInput: null,
    languageSelect: null,
    loadingDiv: null,
    resultDiv: null,
    errorDiv: null,
    postContent: null,
    errorMessage: null,
    copyButton: null
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    setupEventListeners();
    setupExampleTopics();
});

function initializeElements() {
    elements.form = document.getElementById('postForm');
    elements.topicInput = document.getElementById('topic');
    elements.languageSelect = document.getElementById('language');
    elements.loadingDiv = document.getElementById('loading');
    elements.resultDiv = document.getElementById('result');
    elements.errorDiv = document.getElementById('error');
    elements.postContent = document.getElementById('postContent');
    elements.errorMessage = document.getElementById('errorMessage');
    elements.copyButton = document.querySelector('.copy-button');
}

function setupEventListeners() {
    // Form submission
    elements.form.addEventListener('submit', handleFormSubmit);
    
    // Topic input enhancements
    elements.topicInput.addEventListener('focus', handleTopicFocus);
    elements.topicInput.addEventListener('input', handleTopicInput);
    
    // Language change
    elements.languageSelect.addEventListener('change', handleLanguageChange);
    
    // Copy button
    if (elements.copyButton) {
        elements.copyButton.addEventListener('click', copyToClipboard);
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

function setupExampleTopics() {
    elements.topicInput.addEventListener('focus', function() {
        if (!this.value) {
            const randomExample = CONFIG.EXAMPLES[Math.floor(Math.random() * CONFIG.EXAMPLES.length)];
            this.placeholder = `e.g., ${randomExample}`;
        }
    });
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const topic = elements.topicInput.value.trim();
    const language = elements.languageSelect.value;
    
    if (!validateInput(topic)) {
        return;
    }
    
    try {
        showLoading();
        const response = await generatePost(topic, language);
        displayResult(response);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function validateInput(topic) {
    if (!topic) {
        showError('Please enter a topic');
        elements.topicInput.focus();
        return false;
    }
    
    if (topic.length < 2) {
        showError('Topic must be at least 2 characters long');
        elements.topicInput.focus();
        return false;
    }
    
    if (topic.length > 100) {
        showError('Topic must be less than 100 characters');
        elements.topicInput.focus();
        return false;
    }
    
    return true;
}

async function generatePost(topic, language) {
    const response = await fetch(CONFIG.API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic, language })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
        throw new Error(data.error || 'An error occurred while generating the post');
    }
    
    return data;
}

function displayResult(data) {
    elements.postContent.textContent = data.content;
    elements.resultDiv.style.display = 'block';
    
    // Remove any existing demo notices
    const existingNotices = elements.resultDiv.querySelectorAll('.demo-notice');
    existingNotices.forEach(notice => notice.remove());
    
    // Add demo mode indicator if applicable
    if (data.demo_mode) {
        const demoIndicator = document.createElement('div');
        demoIndicator.className = 'demo-notice';
        demoIndicator.innerHTML = '<strong>Demo Mode:</strong> This is a sample post. Connect your OpenAI API for AI-generated content.';
        elements.resultDiv.insertBefore(demoIndicator, elements.postContent);
    }
    
    // Scroll to result
    elements.resultDiv.scrollIntoView({ behavior: 'smooth' });
    
    // Analytics (if needed)
    trackGeneration(data.topic, data.language);
}

function showLoading() {
    elements.loadingDiv.style.display = 'block';
    elements.resultDiv.style.display = 'none';
    elements.errorDiv.style.display = 'none';
}

function hideLoading() {
    elements.loadingDiv.style.display = 'none';
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorDiv.style.display = 'block';
    elements.resultDiv.style.display = 'none';
    
    // Auto-hide error after 5 seconds
    setTimeout(() => {
        elements.errorDiv.style.display = 'none';
    }, 5000);
}

function copyToClipboard() {
    const content = elements.postContent.textContent;
    
    if (!content) {
        showError('No content to copy');
        return;
    }
    
    if (navigator.clipboard) {
        navigator.clipboard.writeText(content).then(() => {
            showCopySuccess();
        }).catch(err => {
            console.error('Failed to copy with navigator.clipboard:', err);
            fallbackCopyTextToClipboard(content);
        });
    } else {
        fallbackCopyTextToClipboard(content);
    }
}

function fallbackCopyTextToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.top = '0';
    textArea.style.left = '0';
    textArea.style.width = '2em';
    textArea.style.height = '2em';
    textArea.style.padding = '0';
    textArea.style.border = 'none';
    textArea.style.outline = 'none';
    textArea.style.boxShadow = 'none';
    textArea.style.background = 'transparent';
    
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        const successful = document.execCommand('copy');
        if (successful) {
            showCopySuccess();
        } else {
            throw new Error('Copy command was unsuccessful');
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
        showError('Failed to copy text. Please select and copy manually.');
    }
    
    document.body.removeChild(textArea);
}

function showCopySuccess() {
    const originalText = elements.copyButton.textContent;
    const originalBackground = elements.copyButton.style.background;
    
    elements.copyButton.textContent = '✅ Copied!';
    elements.copyButton.style.background = '#28a745';
    
    setTimeout(() => {
        elements.copyButton.textContent = originalText;
        elements.copyButton.style.background = originalBackground;
    }, CONFIG.COPY_SUCCESS_DURATION);
}

function handleTopicFocus() {
    // Clear any existing errors when focusing on topic input
    elements.errorDiv.style.display = 'none';
}

function handleTopicInput() {
    // Real-time validation feedback
    const topic = elements.topicInput.value.trim();
    
    if (topic.length > 100) {
        elements.topicInput.style.borderColor = '#dc3545';
    } else if (topic.length > 0) {
        elements.topicInput.style.borderColor = '#28a745';
    } else {
        elements.topicInput.style.borderColor = '#ddd';
    }
}

function handleLanguageChange() {
    // Optional: Add language-specific behavior
    const selectedLanguage = elements.languageSelect.value;
    console.log('Language changed to:', selectedLanguage);
}

function handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        elements.form.dispatchEvent(new Event('submit'));
    }
    
    // Escape to close error messages
    if (e.key === 'Escape') {
        elements.errorDiv.style.display = 'none';
    }
}

function trackGeneration(topic, language) {
    // Optional: Add analytics tracking
    console.log('Post generated:', { topic, language, timestamp: new Date().toISOString() });
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function sanitizeInput(input) {
    return input.replace(/[<>]/g, '');
}

// Export functions for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateInput,
        sanitizeInput,
        debounce
    };
}