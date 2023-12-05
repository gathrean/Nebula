function displayPredictedLabels(labelsData) {
    const labelsContainer = document.getElementById('predictedLabels');
    labelsContainer.innerHTML = '';

    let labelsString = '';
    if (Array.isArray(labelsData)) {
        labelsString = labelsData.join(',');
    } else if (typeof labelsData === 'string') {
        labelsString = labelsData;
    } else {
        console.error('Invalid labels data type:', labelsData);
        return;
    }

    const labelParts = labelsString.split(',').filter(part => part.includes(':'));
    
    // Loop through all elements in labelParts
    for (let i = 0; i < labelParts.length; i++) {
        const part = labelParts[i];
        const [instrument, probability] = part.split(':').map(s => s.trim());
        
        // Check if instrument and probability are valid
        if (instrument && probability) {
            const probabilityPercentage = (parseFloat(probability) * 100).toFixed(2) + '%';
            const labelElement = document.createElement('div');
            labelElement.textContent = `${instrument}`;
            labelsContainer.appendChild(labelElement);
        }
    }
}
