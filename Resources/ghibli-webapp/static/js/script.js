let currentFile = null;

document.getElementById('fileInput').addEventListener('change', function(e) {
    if (this.files && this.files[0]) {
        handleFileSelect(this.files[0]);
    }
});

function handleFileSelect(file) {
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
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
    
    try {
        // Show loading section
        document.getElementById('loadingSection').style.display = 'block';
        document.querySelector('.upload-section').style.display = 'none';
        document.querySelector('.info-section').style.display = 'none';
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Start polling for status
        pollStatus(data.conversion_id);
        
    } catch (error) {
        alert('Upload failed: ' + error.message);
        document.getElementById('loadingSection').style.display = 'none';
        document.querySelector('.upload-section').style.display = 'block';
        document.querySelector('.info-section').style.display = 'grid';
    }
}

async function pollStatus(conversionId) {
    let attempts = 0;
    const maxAttempts = 180; // 15 minutes max
    
    const poll = async () => {
        attempts++;
        
        if (attempts > maxAttempts) {
            window.location.href = `/result/${conversionId}`;
            return;
        }
        
        try {
            const response = await fetch(`/status/${conversionId}`);
            const status = await response.json();
            
            // Update progress
            document.getElementById('statusMessage').textContent = status.message;
            
            // Simple progress simulation
            const progress = Math.min(90, (attempts / maxAttempts) * 100);
            document.getElementById('progressFill').style.width = progress + '%';
            
            if (status.status === 'completed' || status.status === 'error') {
                window.location.href = `/result/${conversionId}`;
            } else {
                // Continue polling
                setTimeout(poll, 2000);
            }
            
        } catch (error) {
            console.error('Status polling error:', error);
            setTimeout(poll, 5000);
        }
    };
    
    poll();
}

// Drag and drop support
document.addEventListener('DOMContentLoaded', function() {
    const uploadBox = document.getElementById('uploadBox');
    
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
});