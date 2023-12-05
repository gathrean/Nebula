// The file input
const fileInput = document.getElementById('audioFile');
let isDraggingFile = false;

// Event listener for the user clicks on the box
dropZone.addEventListener('click', () => fileInput.click());

// Event listener for when the user drags a file over the box
document.body.addEventListener('dragover', (event) => {
    event.stopPropagation();
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';

    if (!isDraggingFile) {
        isDraggingFile = true;
        showOverlay();
    }
});

// Event listener for when the user drags a file out of the box
document.body.addEventListener('dragleave', (event) => {
    event.stopPropagation();
    event.preventDefault();

    const dragLeaveTarget = event.relatedTarget || event.toElement;
    if (!dragLeaveTarget || dragLeaveTarget.nodeName === 'HTML') {
        isDraggingFile = false;
        hideOverlay();
    }
});

// Event listener for when the user drops a file on the box
document.body.addEventListener('drop', (event) => {
    event.stopPropagation();
    event.preventDefault();
    isDraggingFile = false;
    hideOverlay();

    const files = event.dataTransfer.files;
    handleFileUpload(files[0]);
});

// Event listener for when the mouse leaves the window entirely
document.addEventListener('mouseleave', () => {
    if (isDraggingFile) {
        isDraggingFile = false;
        hideOverlay();
    }
});

// Function to show the overlay
function showOverlay() {
    const overlay = document.getElementById('overlay');
    overlay.style.display = 'block';
}

// Function to hide the overlay
function hideOverlay() {
    const overlay = document.getElementById('overlay');
    overlay.style.display = 'none';
}

// Event listener for when the user selects a file
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFileUpload(fileInput.files[0]);
    } else {
        // Show 'None' image if no file selected
        const nebulaNone = document.querySelector('.character-container:nth-child(1)');
        const nebulaDone = document.querySelector('.character-container:nth-child(3)');
        
        nebulaNone.style.display = 'block';
        nebulaDone.style.display = 'none';
    }
});
