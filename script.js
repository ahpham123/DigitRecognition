const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const classifyBtn = document.getElementById('classifyBtn');
const clearBtn = document.getElementById('clearBtn');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');

const GRID_SIZE = 28;
const PIXEL_SIZE = canvas.width / GRID_SIZE;
let pixels = Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(0));

let isDrawing = false;

const API_URL = 'http://localhost:5000/predict';

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.imageSmoothingEnabled = false;

function drawGrid() {
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * PIXEL_SIZE, 0);
        ctx.lineTo(i * PIXEL_SIZE, canvas.height);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, i * PIXEL_SIZE);
        ctx.lineTo(canvas.width, i * PIXEL_SIZE);
        ctx.stroke();
    }
}

function getGridPosition(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    
    const gridX = Math.floor(x / PIXEL_SIZE);
    const gridY = Math.floor(y / PIXEL_SIZE);
    
    return { gridX, gridY };
}

function fillPixel(gridX, gridY, intensity = 255) {
    if (gridX < 0 || gridX >= GRID_SIZE || gridY < 0 || gridY >= GRID_SIZE) return;
    
    pixels[gridY][gridX] = Math.min(255, pixels[gridY][gridX] + intensity);
    const pixelValue = pixels[gridY][gridX];
    ctx.fillStyle = `rgb(${pixelValue}, ${pixelValue}, ${pixelValue})`;
    ctx.fillRect(gridX * PIXEL_SIZE, gridY * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
}

function fillPixelWithBrush(gridX, gridY) {
    fillPixel(gridX, gridY, 255);
    
    // Brush effect
    const neighbors = [
        {x: gridX-1, y: gridY, intensity: 80},
        {x: gridX+1, y: gridY, intensity: 80},
        {x: gridX, y: gridY-1, intensity: 80},
        {x: gridX, y: gridY+1, intensity: 80},
        {x: gridX-1, y: gridY-1, intensity: 40},
        {x: gridX+1, y: gridY-1, intensity: 40},
        {x: gridX-1, y: gridY+1, intensity: 40},
        {x: gridX+1, y: gridY+1, intensity: 40}
    ];
    
    neighbors.forEach(n => {
        if (n.x >= 0 && n.x < GRID_SIZE && n.y >= 0 && n.y < GRID_SIZE) {
            fillPixel(n.x, n.y, n.intensity);
        }
    });
}

function startDrawing(e) {
    isDrawing = true;
    const { gridX, gridY } = getGridPosition(e);
    fillPixelWithBrush(gridX, gridY);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const { gridX, gridY } = getGridPosition(e);
    fillPixelWithBrush(gridX, gridY);
}

function stopDrawing() {
    isDrawing = false;
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', () => {
    pixels = Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(0));
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGrid();
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
});

classifyBtn.addEventListener('click', async () => {
    try {
        resultDiv.classList.add('hidden');
        errorDiv.classList.add('hidden');
        loadingDiv.classList.remove('hidden');
        classifyBtn.disabled = true;
        
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pixels: pixels })
        });
        
        if (!response.ok) {
            throw new Error('Failed to classify image');
        }
        
        const data = await response.json();
        
        document.getElementById('predictionValue').textContent = data.prediction;
        document.getElementById('confidenceValue').textContent = 
            (data.confidence * 100).toFixed(2) + '%';
        
        const probContainer = document.getElementById('probabilitiesContainer');
        probContainer.innerHTML = '';
        
        for (let i = 0; i < 10; i++) {
            const prob = data.probabilities[i] * 100;
            const item = document.createElement('div');
            item.className = 'prob-item';
            item.innerHTML = `
                <span class="prob-label">${i}:</span>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${prob}%"></div>
                </div>
                <span class="prob-value">${prob.toFixed(1)}%</span>
            `;
            probContainer.appendChild(item);
        }
        
        resultDiv.classList.remove('hidden');
        
    } catch (error) {
        errorDiv.textContent = 'Error: ' + error.message + 
            '. Make sure the backend server is running!';
        errorDiv.classList.remove('hidden');
    } finally {
        loadingDiv.classList.add('hidden');
        classifyBtn.disabled = false;
    }
});        
drawGrid();