let currentFile = null;

// File input handler
document.getElementById('fileInput').addEventListener('change', function(e) {
    if (this.files && this.files[0]) {
        handleFileSelect(this.files[0]);
    }
});

function handleFileSelect(file) {
    // Basic file validation
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }
    
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        alert('Please select a valid image file (PNG, JPG, JPEG, WEBP, BMP)');
        return;
    }
    
    currentFile = file;
    
    // Update UI
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileInfo').style.display = 'block';
    document.getElementById('uploadBox').style.display = 'none';
}

function removeFile() {
    currentFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('uploadBox').style.display = 'block';
}

async function startConversion() {
    if (!currentFile) {
        alert('Please select a file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    // Get prompt if available
    const promptInput = document.getElementById('promptInput');
    if (promptInput && promptInput.value.trim()) {
        formData.append('prompt', promptInput.value.trim());
    }
    
    try {
        // Show loading section
        document.getElementById('loadingSection').style.display = 'block';
        document.querySelector('.upload-section').style.display = 'none';
        document.querySelector('.info-section').style.display = 'none';
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Upload failed: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Start polling for status
        pollStatus(data.conversion_id);
        
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Upload failed: ' + error.message);
        resetUI();
    }
}

async function pollStatus(conversionId) {
    let attempts = 0;
    const maxAttempts = 600; // 20 minutes max (for SDXL processing)
    
    const poll = async () => {
        attempts++;
        
        if (attempts > maxAttempts) {
            window.location.href = `/result/${conversionId}`;
            return;
        }
        
        try {
            const response = await fetch(`/status/${conversionId}`);
            
            if (!response.ok) {
                throw new Error(`Status check failed: ${response.status}`);
            }
            
            const status = await response.json();
            
            // Update progress
            document.getElementById('statusMessage').textContent = status.message;
            
            // Update progress bar based on status message
            updateProgressBar(status.message, attempts, maxAttempts);
            
            if (status.status === 'completed' || status.status === 'error') {
                window.location.href = `/result/${conversionId}`;
            } else {
                // Continue polling
                setTimeout(poll, 3000); // Check every 3 seconds
            }
            
        } catch (error) {
            console.error('Status polling error:', error);
            // Show error but keep trying
            document.getElementById('statusMessage').textContent = 'Connection issue, retrying...';
            setTimeout(poll, 5000);
        }
    };
    
    poll();
}

function updateProgressBar(message, attempts, maxAttempts) {
    const progressFill = document.getElementById('progressFill');
    let progress = 0;
    
    // Estimate progress based on status message
    if (message.includes('Starting conversion') || message.includes('Loading image')) {
        progress = 10;
    } else if (message.includes('Enhancing background')) {
        progress = 30;
    } else if (message.includes('Full image transformation')) {
        progress = 50;
    } else if (message.includes('Generating')) {
        progress = 70 + (attempts / maxAttempts) * 20;
    } else if (message.includes('Final quality enhancement')) {
        progress = 90;
    } else {
        // Fallback: time-based progress
        progress = Math.min(90, (attempts / maxAttempts) * 100);
    }
    
    progressFill.style.width = progress + '%';
}

function resetUI() {
    document.getElementById('loadingSection').style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    document.querySelector('.info-section').style.display = 'grid';
}

// Drag and drop support
document.addEventListener('DOMContentLoaded', function() {
    const uploadBox = document.getElementById('uploadBox');
    
    if (uploadBox) {
        uploadBox.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#764ba2';
            this.style.background = '#f0f3ff';
        });
        
        uploadBox.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '#667eea';
            this.style.background = 'white';
        });
        
        uploadBox.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#667eea';
            this.style.background = 'white';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });
    }
});

// Enter key support for prompt textarea
document.addEventListener('DOMContentLoaded', function() {
    const promptInput = document.getElementById('promptInput');
    if (promptInput) {
        promptInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                // Ctrl+Enter to start conversion
                startConversion();
            }
        });
    }
});

// Add some sample prompts for user convenience
const samplePrompts = [
    "Studio Ghibli style, magical forest, vibrant colors",
    "Anime style transformation, dreamy atmosphere",
    "Ghibli background, detailed environment, pastel colors",
    "Beautiful anime scene, soft lighting, cinematic"
];

// Add sample prompt button if needed
function addSamplePrompt() {
    const promptInput = document.getElementById('promptInput');
    if (promptInput) {
        const randomPrompt = samplePrompts[Math.floor(Math.random() * samplePrompts.length)];
        promptInput.value = randomPrompt;
    }
}