let modeloGlobal = null;

// Función para cargar los datos
async function cargarDatos() {
    try {
        console.log('Iniciando carga de datos...');
        
        // Cargar los archivos JSON
        const xRaw = await fetch('/static/data/x_train.json').then(r => r.json());
        const yRaw = await fetch('/static/data/y_train.json').then(r => r.json());
        
        console.log('Datos cargados:');
        console.log('Primeras 5 imágenes:', xRaw.slice(0, 5)); 
        console.log('Primeras 5 etiquetas:', yRaw.slice(0, 5)); 

        // Convertir las etiquetas a tipo int32
        const ys = tf.tensor1d(yRaw, 'int32').oneHot(10); // Etiquetas (0 a 9), convertidas a 'int32'

        // Convertir los datos de las imágenes en tensores de TensorFlow.js
        const xs = tf.tensor4d(xRaw, [xRaw.length, 28, 28, 1]); // Imágenes

        return { xs, ys };
    } catch (error) {
        console.error("Error al cargar los datos:", error);
        document.getElementById('salida').innerText = 'Error al cargar los datos: ' + error;
    }
}

// Función para entrenar el modelo
async function entrenar() {
    document.getElementById('salida').innerText = 'Cargando datos...';
    const { xs, ys } = await cargarDatos();
    if (!xs || !ys) return;

    const modelo = tf.sequential();
    modelo.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
    modelo.add(tf.layers.dense({ units: 60, activation: 'relu' }));
    modelo.add(tf.layers.dense({ units: 60, activation: 'relu' }));
    modelo.add(tf.layers.dense({ units: 60, activation: 'relu' }));
    modelo.add(tf.layers.dense({ units: 60, activation: 'relu' }));
    modelo.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    modelo.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    document.getElementById('salida').innerText = 'Entrenando...';
    await modelo.fit(xs, ys, {
        epochs: 100,
        batchSize: 32,
        callbacks: {
            onEpochEnd: (ep, logs) => {
                console.log(logs);
                document.getElementById('salida').innerText =
                    `Epoch ${ep + 1}: accuracy = ${logs.acc.toFixed(3)}`;
            }
        }
    });

    modeloGlobal = modelo; // Guardamos para usar en la predicción
    document.getElementById('salida').innerText += '\n✅ Entrenamiento completo.';
}
// =====================
// Canvas para dibujar
// =====================
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let dibujando = false;

canvas.addEventListener('mousedown', () => dibujando = true);
canvas.addEventListener('mouseup', () => dibujando = false);
canvas.addEventListener('mousemove', dibujar);

function dibujar(event) {
    if (!dibujando) return;
    const x = event.offsetX;
    const y = event.offsetY;
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, 2 * Math.PI);
    ctx.fill();
}

function limpiarCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
limpiarCanvas(); // Fondo inicial negro

// ==========================
// Predicción desde dibujo
// ==========================
async function predecirDibujo() {
    if (!modeloGlobal) {
        alert("Primero entrena el modelo.");
        return;
    }

    const imagenTensor = tf.tidy(() => {
        let imageData = ctx.getImageData(0, 0, 280, 280);
        let img = tf.browser.fromPixels(imageData, 1); // blanco y negro
        img = tf.image.resizeBilinear(img, [28, 28]);
        img = img.div(255);
        return img.expandDims(0); // [1, 28, 28, 1]
    });

    const prediccion = modeloGlobal.predict(imagenTensor);
    const resultado = prediccion.argMax(1);
    const valor = (await resultado.data())[0];

    document.getElementById('prediccion').innerText = `🔍 Predicción: ${valor}`;
    tf.dispose([imagenTensor, prediccion, resultado]);
}