import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton")
const logButton = document.getElementById("logButton")
const saveGesture = document.getElementById("saveGesture");
const classifyButton = document.getElementById("classifyButton")

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

const drawUtils = new DrawingUtils(canvasCtx)
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let nn;
let loggedGestures = [];
let gesture = [];
const labelInput = document.getElementById("labelInput");
labelInput.addEventListener("input", () => {
    label = labelInput.value.trim().toLowerCase();
});

let label;


function createNeuralNetwork() {
    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({
        task: 'classification',
        debug: true
    });

    const options = {
        model: "./model/model.json",
        metadata: "./model/model_meta.json",
        weights: "./model/model.weights.bin"
    }

    nn.load(options, createHandLandmarker())
}


/********************************************************************
 // CREATE THE POSE DETECTOR
 ********************************************************************/
const createHandLandmarker = async () => {
    console.log("neural model is loaded!")

    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")

    enableWebcamButton.addEventListener("click", (e) => enableCam(e))
    logButton.addEventListener("click", (e) => logAllHands(e))
    saveGesture.addEventListener("click", (e) => saveGestureData(e))
    classifyButton.addEventListener("click", (e) => classifyHand(e))
}

/********************************************************************
 // START THE WEBCAM
 ********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
 // START PREDICTIONS
 ********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for(let hand of results.landmarks){
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam)
    }
}

/********************************************************************
 // LOG HAND COORDINATES IN THE CONSOLE
 ********************************************************************/
function logAllHands() {
    if (!results || !results.landmarks || results.landmarks.length === 0) {
        console.warn("No gesture detected to log.");
        return;
    }

    gesture = results.landmarks[0];

    if (!Array.isArray(gesture) || gesture.length !== 21) {
        console.warn("Expected 21 landmarks for a complete gesture.");
        return;
    }

    // Flatten each landmark into [x, y, z], then flatten all into one big array
    const points = gesture.flatMap(({ x, y, z }) => [x, y, z]);

    loggedGestures.push({
        data: points,
        label: label
    });

    console.log(`Logged gesture for label "${label}" (flattened):`, points);
}

function saveGestureData(){
    if (loggedGestures.length === 0) {
        console.warn("No poses to save.");
        return;
    }

    const blob = new Blob([JSON.stringify(loggedGestures)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "data.json";
    link.click();
    console.log("Pose data saved as data.json");
}

function classifyHand(){
    let numbersonly = [];
    let hand = results.landmarks[0];
    
    for (let point of hand) {
        numbersonly.push(point.x, point.y, point.z);
    }

    nn.classify(numbersonly, (results) => {
        const predicted = parseInt(results[0].label);
        const feedback = document.getElementById("feedback");

        if (predicted === currentAnswer) {
            feedback.innerText = `✅ Klopt! ${predicted} is het juiste antwoord.`;
            feedback.style.color = 'green';
            setTimeout(() => generateSum(), 2000);
        } else {
            feedback.innerText = `❌ ${predicted} is niet goed, probeer het nog eens.`;
            feedback.style.color = 'red';
        }
    });

}

let currentAnswer = null;

function generateSum() {
    const a = Math.floor(Math.random() * 9) + 1; // 1 t/m 9
    const b = Math.floor(Math.random() * (10 - a)) + 1; // zodat a + b max 10 is
    currentAnswer = a + b;

    document.getElementById("question").innerText = `Wat is ${a} + ${b}?`;
    document.getElementById("feedback").innerText = "";
}

/********************************************************************
 // START THE APP
 ********************************************************************/
createNeuralNetwork();
generateSum();