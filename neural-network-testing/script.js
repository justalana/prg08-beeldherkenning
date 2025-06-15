import testData from './data.json' with {type: 'json'};

let nn;
let i = 0;
let correct = 0;

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

    nn.load(options, startTesting)
}

function startTesting() {
    document.getElementById("accuracyText").innerText = "Model wordt getest, even geduld...";

    let testGesture = testData[i];
    nn.classify(testGesture.data, (results) => {
        if (results[0].label === testGesture.label) {
            console.log(`${testGesture.label} is correct`);
            correct++;
        }

        i++;

        if (i < testData.length) {
            startTesting();
        } else {
            let accuracy = (correct/testData.length) * 100;
            document.getElementById("accuracyText").innerText = `Model is klaar! Accuratesse: ${accuracy.toFixed(1)}% (${correct}/${testData.length} goed).`;
        }
    })
}

createNeuralNetwork()