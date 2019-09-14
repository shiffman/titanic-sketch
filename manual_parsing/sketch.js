/////////////////////////////////////////////////////
//                                                 //
//     Neural Network Training with the Titanic    //
//                                                 //
/////////////////////////////////////////////////////

// By: Lydia Jessup
// Adjustments by Dan Shiffman
// Date: Sept 13, 2019
// Description: Using TF.js to train an neural network to predict survival on the titanic
// Also:  This is heavily based off of examples by Dan Shiffman and the tf.js team

///////////////////////////////////////////////
// Import, normalize and transform data
///////////////////////////////////////////////

// Paths
const TRAIN_DATA_PATH = './titanic_train.csv';
const TEST_DATA_PATH = './titanic_test.csv';

// have to put in all min and max values in order to normalize
// age and fare -- for class and sex will do one hot encoding
// I got these values from before when I was cleaning the data
const AGE_MIN = 0.0;
const AGE_MAX = 80.0;
const FARE_MIN = 0.0;
const FARE_MAX = 512.0;

let model;
let trainingData = {};
let rawData;

// Load the raw data from CSV
function preload() {
  rawData = loadStrings(TRAIN_DATA_PATH);
}

// Three step process
function setup() {
  parseData();
  createModel();
  train();
}

// Manual parsing
function parseData() {
  // Put data from CSV into plain arrays
  let xs = [];
  let ys = [];
  // Going through one row at a time very manually
  // Could use higher order array functions but keeping things simple
  for (let i = 1; i < rawData.length; i++) {
    let row = rawData[i].split(',');
    // Normalize age
    row[0] = map(parseFloat(row[0]), AGE_MIN, AGE_MAX, 0, 1);
    // Normalize fare
    row[1] = map(parseFloat(row[1]), FARE_MIN, FARE_MAX, 0, 1);
    // Just convert to number
    row[2] = parseFloat(row[2]);

    // Add to new array
    xs.push(row.slice(0, 3));
    ys.push(row.slice(3, 4)[0]);
  }

  // Convert to tensors
  trainingData.xs = tf.tensor(xs);
  // One hot encoding for categorical output
  trainingData.ys = tf.oneHot(tf.tensor1d(ys, 'int32'), 2);
}

// Create a model, very arbitrary architecture
function createModel() {
  model = tf.sequential();
  const hidden = tf.layers.dense({
    units: 16,
    inputShape: [3],
    activation: 'relu'
  });
  const output = tf.layers.dense({
    units: 2,
    activation: 'softmax'
  });
  model.add(hidden);
  model.add(output);

  model.compile({
    optimizer: tf.train.adam(),
    // This loss breaks things?
    // loss: 'sparseCategoricalCrossentropy',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
}

// Tran model
async function train() {
  await model.fit(trainingData.xs, trainingData.ys, {
    shuffle: true,
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch: ${epoch} - loss: ${logs.loss.toFixed(3)}`);
      }
    },
  });
}


