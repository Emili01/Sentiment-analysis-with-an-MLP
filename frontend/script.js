// DOM Elements
const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const resultCard = document.getElementById('result-card');
const loadingElement = document.getElementById('loading');
const errorElement = document.getElementById('error');
const errorMessage = document.getElementById('error-message');
const sentimentLabel = document.getElementById('sentiment-label');
const sentimentIcon = document.getElementById('sentiment-icon');
const analyzedTextContent = document.getElementById('analyzed-text-content');

// Probability bars
const negativeBar = document.getElementById('negative-bar');
const neutralBar = document.getElementById('neutral-bar');
const positiveBar = document.getElementById('positive-bar');
const negativeProb = document.getElementById('negative-prob');
const neutralProb = document.getElementById('neutral-prob');
const positiveProb = document.getElementById('positive-prob');

// API Configuration
// Use /api/ to leverage the reverse proxy configured in Nginx
const API_URL = '/api';
const PREDICT_ENDPOINT = `${API_URL}/predict`;

// Emojis for each sentiment
const sentimentEmojis = {
    "Negative": "😢",
    "Neutral": "😐",
    "Positive": "😊"
};

// Colors for each sentiment
const sentimentClasses = {
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive"
};

// Initialization
document.addEventListener('DOMContentLoaded', function() {
    // Check server status when page loads
    checkServerStatus();
    
    // Add event to analyze button
    analyzeBtn.addEventListener('click', analyzeSentiment);
    
    // Allow submit when pressing Enter in the textarea
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            analyzeSentiment();
        }
    });
});

// Check server status
function checkServerStatus() {
    fetch(`${API_URL}/health`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'degraded') {
                showMessage('The server is in degraded mode. Some features may not be available.', 'warning');
            }
        })
        .catch(error => {
            console.error('Error checking server status:', error);
            showMessage('Could not connect to the analysis server.', 'error');
        });
}

// Main function to analyze sentiment
function analyzeSentiment() {
    const text = textInput.value.trim();
    
    // Validate input
    if (!text) {
        showError('Please enter a text to analyze.');
        return;
    }
    
    // Show loading status
    showLoading();
    
    // API request
    fetch(PREDICT_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw err; });
        }
        return response.json();
    })
    .then(result => {
        displayResult(result);
    })
    .catch(error => {
        console.error('Error:', error);
        showError(error.detail || 'An error occurred while analyzing the text.');
    });
}

// Display analysis result
function displayResult(result) {
    // Hide loading and error
    hideLoading();
    hideError();
    
    // Show analyzed text
    analyzedTextContent.textContent = result.text;
    
    // Update sentiment label
    sentimentLabel.textContent = result.sentiment_label;
    
    // Update emoji based on sentiment
    const emojiElement = sentimentIcon.querySelector('.emoji');
    emojiElement.textContent = sentimentEmojis[result.sentiment_label] || "😐";
    
    // Update CSS class of icon
    sentimentIcon.className = 'sentiment-icon';
    sentimentIcon.classList.add(sentimentClasses[result.sentiment_label] || 'neutral');
    
    // Update probability bars
    const probabilities = result.probabilities;
    
    // Convert to percentages
    const negativePercent = Math.round(probabilities[0] * 100);
    const neutralPercent = Math.round(probabilities[1] * 100);
    const positivePercent = Math.round(probabilities[2] * 100);
    
    // Animate progress bars
    setTimeout(() => {
        negativeBar.style.width = `${negativePercent}%`;
        neutralBar.style.width = `${neutralPercent}%`;
        positiveBar.style.width = `${positivePercent}%`;
        
        negativeProb.textContent = `${negativePercent}%`;
        neutralProb.textContent = `${neutralPercent}%`;
        positiveProb.textContent = `${positivePercent}%`;
    }, 100);
    
    // Show result card
    resultCard.classList.remove('hidden');
}

// UI Functions
function showLoading() {
    resultCard.classList.add('hidden');
    errorElement.classList.add('hidden');
    loadingElement.classList.remove('hidden');
}

function hideLoading() {
    loadingElement.classList.add('hidden');
}

function showError(message) {
    resultCard.classList.add('hidden');
    loadingElement.classList.add('hidden');
    
    errorMessage.textContent = message;
    errorElement.classList.remove('hidden');
}

function hideError() {
    errorElement.classList.add('hidden');
}

// Show temporary message
function showMessage(message, type = 'info') {
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = `message ${type}`;
    messageElement.textContent = message;
    
    // Add to DOM
    document.querySelector('.container').appendChild(messageElement);
    
    // Remove after 5 seconds
    setTimeout(() => {
        messageElement.classList.add('fade-out');
        setTimeout(() => {
            messageElement.remove();
        }, 500);
    }, 5000);
} 