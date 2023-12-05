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

    // Skip the first and the last elements of the array
    for (let i = 1; i < labelParts.length - 1; i++) {
        const part = labelParts[i];
        const [instrument, probability] = part.split(':').map(s => s.trim());
        const probabilityPercentage = (parseFloat(probability) * 100).toFixed(2) + '%';
        const labelElement = document.createElement('div');
        labelElement.textContent = `${instrument}: ${probabilityPercentage}`;
        labelsContainer.appendChild(labelElement);
    }
}
