document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const loader = document.getElementById('loader');
    const errorContainer = document.getElementById('error-container');
    const errorText = document.getElementById('error-text');
    const uploadCounter = document.getElementById('upload-counter');

    let processedCount = 0;

    // --- Event Listeners ---

    // Trigger file input when the custom button is clicked
    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // Show image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(file);

            // Reset UI and trigger prediction
            resetUI();
            predict(file);
        }
    });

    // --- UI Helper Functions ---

    function showLoader() {
        loader.classList.remove('hidden');
    }

    function hideLoader() {
        loader.classList.add('hidden');
    }

    function showResult(predictedClass) {
        resultText.innerHTML = `BREED: <span class="highlight">${predictedClass}</span>`;
        resultContainer.classList.remove('hidden');

        // Add to recent detections log
        const detectionLog = document.getElementById('detection-log');
        const logPlaceholder = document.querySelector('.log-placeholder');
        if (logPlaceholder) {
            logPlaceholder.remove();
        }
        const newEntry = document.createElement('p');
        const timestamp = new Date().toISOString();
        newEntry.innerHTML = `[${timestamp}] <span class="log-highlight">DETECTED:</span> ${predictedClass}`;
        detectionLog.prepend(newEntry);
    }

    function showError(message) {
        errorText.textContent = `ERROR // ${message}`;
        errorContainer.classList.remove('hidden');
    }

    function resetUI() {
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        hideLoader();
    }

    // --- API Call ---

    async function predict(file) {
        showLoader();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            hideLoader();

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            if (data.predictions && data.predictions.length > 0) {
                showResult(data.predictions[0].label);
            } else {
                showError("No predictions returned from model.");
            }

            // Increment and update the counter
            processedCount++;
            uploadCounter.textContent = processedCount;

        } catch (error) {
            hideLoader();
            showError(error.message);
        }
    }
});

// Add a highlight style dynamically to avoid cluttering CSS
const style = document.createElement('style');
style.innerHTML = `
.highlight {
    color: #ff0000;
    font-weight: 900;
}
`;
document.head.appendChild(style);
