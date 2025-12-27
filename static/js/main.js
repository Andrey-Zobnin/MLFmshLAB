// Main JavaScript for Math Equation Recognition and Solver App

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('upload-button');
    const submitButton = document.getElementById('submit-button');
    const uploadContainer = document.querySelector('.upload-container');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const uploadPrompt = document.getElementById('upload-prompt');
    const removeImageBtn = document.getElementById('remove-image');
    const resultsContainer = document.getElementById('results-container');
    const predictionsList = document.getElementById('predictions-list');
    const loadingElement = document.getElementById('loading');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const themeToggle = document.getElementById('themeToggle');

    // File handling
    let selectedFile = null;

    // Event listeners
    uploadButton.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelection);
    removeImageBtn.addEventListener('click', removeImage);
    submitButton.addEventListener('click', analyzeImage);
    
    // Drag and drop functionality
    uploadContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadContainer.classList.add('drag-over');
    });
    
    uploadContainer.addEventListener('dragleave', () => {
        uploadContainer.classList.remove('drag-over');
    });
    
    uploadContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadContainer.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelection();
        }
    });
    
    uploadContainer.addEventListener('click', () => {
        if (previewContainer.classList.contains('d-none')) {
            fileInput.click();
        }
    });

    // Theme stylesheet update - executed only if theme-stylesheet exists
    const themeStylesheet = document.getElementById('theme-stylesheet');
    if (themeStylesheet) {
        function updateThemeStylesheet(theme) {
            if (theme === 'light') {
                themeStylesheet.href = 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css';
            } else {
                themeStylesheet.href = 'https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css';
            }
        }
        
        // Update theme when page loads
        const currentTheme = document.body.getAttribute('data-bs-theme');
        updateThemeStylesheet(currentTheme);
    }

    // Handle file selection
    function handleFileSelection() {
        if (fileInput.files && fileInput.files[0]) {
            selectedFile = fileInput.files[0];
            
            // Check if it's an image
            if (!selectedFile.type.match('image.*')) {
                const errorText = document.documentElement.lang === 'ru' ? 
                    'Пожалуйста, выберите файл изображения (JPG, PNG и т.д.)' : 
                    'Please select an image file (JPG, PNG, etc.)';
                showError(errorText);
                return;
            }
            
            // Create URL for preview
            const objectUrl = URL.createObjectURL(selectedFile);
            imagePreview.src = objectUrl;
            
            // Show preview, hide upload prompt
            previewContainer.classList.remove('d-none');
            uploadPrompt.classList.add('d-none');
            
            // Enable submit button
            submitButton.disabled = false;
            
            // Clear any previous results and errors
            hideResults();
            hideError();
        }
    }

    // Remove selected image
    function removeImage(e) {
        e.stopPropagation();
        selectedFile = null;
        imagePreview.src = '#';
        fileInput.value = '';
        
        // Hide preview, show upload prompt
        previewContainer.classList.add('d-none');
        uploadPrompt.classList.remove('d-none');
        
        // Disable submit button
        submitButton.disabled = true;
        
        // Hide results
        hideResults();
    }

    // Analyze the image
    function analyzeImage() {
        if (!selectedFile) return;
        
        // Show loading state
        showLoading();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Send request to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            hideLoading();
            const errorPrefix = document.documentElement.lang === 'ru' ? 'Ошибка:' : 'Error:';
            showError(`${errorPrefix} ${error.message}`);
        });
    }

    // Display equation and solution results
    function displayResults(data) {
        // Clear previous results
        predictionsList.innerHTML = '';
        
        // Labels for equation and solution based on language
        const equationLabel = document.documentElement.lang === 'ru' ? 'Уравнение' : 'Equation';
        const solutionLabel = document.documentElement.lang === 'ru' ? 'Решение' : 'Solution';
        
        // Create equation result
        const equationItem = document.createElement('li');
        equationItem.className = 'list-group-item prediction-item';
        equationItem.innerHTML = `
            <div class="prediction-confidence bg-primary text-white">
                ${equationLabel}
            </div>
            <div class="prediction-label">
                <div class="prediction-name equation-text">${data.equation}</div>
            </div>
        `;
        
        // Create solution result
        const solutionItem = document.createElement('li');
        solutionItem.className = 'list-group-item prediction-item';
        solutionItem.innerHTML = `
            <div class="prediction-confidence bg-success text-white">
                ${solutionLabel}
            </div>
            <div class="prediction-label">
                <div class="prediction-name solution-text">${data.solution}</div>
                ${data.error ? `<div class="text-danger">${data.error}</div>` : ''}
            </div>
        `;
        
        // Add items to list
        predictionsList.appendChild(equationItem);
        predictionsList.appendChild(solutionItem);
        
        // If there are detailed symbol recognitions, add them
        if (data.details && data.details.recognized_symbols && data.details.recognized_symbols.length > 0) {
            const symbolsLabel = document.documentElement.lang === 'ru' ? 'Распознанные символы' : 'Recognized Symbols';
            
            const symbolsHeader = document.createElement('li');
            symbolsHeader.className = 'list-group-item bg-light';
            symbolsHeader.innerHTML = `<h6 class="mb-0">${symbolsLabel}</h6>`;
            predictionsList.appendChild(symbolsHeader);
            
            // Add each recognized symbol with confidence
            data.details.recognized_symbols.forEach(symbol => {
                const confidence = Math.round(symbol.confidence * 100);
                const colorClass = getConfidenceColorClass(confidence);
                const confidenceText = document.documentElement.lang === 'ru' ? 'уверенность' : 'confidence';
                
                const symbolItem = document.createElement('li');
                symbolItem.className = 'list-group-item prediction-item';
                symbolItem.innerHTML = `
                    <div class="prediction-confidence ${colorClass}" title="${confidence}% ${confidenceText}">
                        ${confidence}%
                    </div>
                    <div class="prediction-label">
                        <div class="prediction-name">${symbol.symbol}</div>
                        <div class="confidence-bar">
                            <div class="confidence-level" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                `;
                
                predictionsList.appendChild(symbolItem);
            });
        }
        
        // Show results container
        resultsContainer.classList.remove('d-none');
    }

    // Get color class based on confidence level
    function getConfidenceColorClass(confidence) {
        if (confidence >= 80) return 'bg-success text-white';
        if (confidence >= 60) return 'bg-info text-white';
        if (confidence >= 40) return 'bg-warning text-dark';
        return 'bg-danger text-white';
    }

    // Show loading state
    function showLoading() {
        loadingElement.classList.remove('d-none');
        resultsContainer.classList.remove('d-none');
        predictionsList.classList.add('d-none');
    }

    // Hide loading state
    function hideLoading() {
        loadingElement.classList.add('d-none');
        predictionsList.classList.remove('d-none');
    }

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorContainer.classList.remove('d-none');
    }

    // Hide error message
    function hideError() {
        errorContainer.classList.add('d-none');
    }

    // Hide results
    function hideResults() {
        resultsContainer.classList.add('d-none');
    }
});
