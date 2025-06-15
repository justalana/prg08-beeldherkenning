import gesturedata from './data.json' with { type: "json" }

let nn
const statusEl = document.getElementById("trainingStatus");

function startTraining(){
    statusEl.innerText = "Data wordt geladen en model wordt getraind...";

    ml5.setBackend('webgl')
    nn = ml5.neuralNetwork({
        task: 'classification',
        debug: true
    });

    for(let gesture of gesturedata) {
        // console.log(gesture)
        nn.addData(gesture.data, {label: gesture.label})
    }

    nn.normalizeData()
    nn.train({epochs: 50}, finishedTraining)
}

function finishedTraining() {
    statusEl.innerText = "Training voltooid! Model is klaar.";

    console.log('finished training!')
    nn.save()

    let demogesture = gesturedata[10].data
    nn.classify(demogesture, (results) => {
        console.log(`I think this pose is a ${results[0].label}`)
        console.log(`I am ${(results[0].confidence.toFixed(2)) * 100}% sure`)
    })
}

startTraining()