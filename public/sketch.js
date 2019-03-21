var trainlabelsbytes;
var trainimagesbytes;
var img;
var trainimages=[];
var trainlabels=[];
var testimages=[];
var testlabels=[];
var trainDataSize=1000;
var testDataSize=200;
var tfmodel;
function preload(){
	trainlabelsbytes=loadBytes("tlabel");
	trainimagesbytes=loadBytes("timage");
}
function createData(){
	var index=16;
	for(var i=0;i<trainDataSize;i++){
		var row=[];
		for(var j=0;j<784;j++){
			row.push(trainimagesbytes.bytes[index]/255);
			index++;
		}
		trainimages.push(row);
	}
	for(var i=0;i<testDataSize;i++){
		var row=[];
		for(var j=0;j<784;j++){
			row.push(trainimagesbytes.bytes[index]/255);
			index++;
		}
		testimages.push(row);	
	}
	index=8;
	for(var i=0;i<trainDataSize;i++){
		var row=[0,0,0,0,0,0,0,0,0,0];
		row[trainlabelsbytes.bytes[index]]=1;
		trainlabels.push(row);
		index++;
	}
	for(var i=0;i<testDataSize;i++){
		var row=[0,0,0,0,0,0,0,0,0,0];
		row[trainlabelsbytes.bytes[index]]=1;
		testlabels.push(row);
		index++;
	}
}
function createtfmodel(){
	tfmodel = tf.sequential();
    tfmodel.add(tf.layers.conv2d({
      inputShape: [28,28,1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }));
    tfmodel.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    }));
    tfmodel.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }));
    tfmodel.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    }));
    tfmodel.add(tf.layers.flatten());
    tfmodel.add(tf.layers.dense({
      units: 10,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    }));
    // const LEARNING_RATE = 0.15;
    // const optimizer = tf.train.sgd(LEARNING_RATE);
    tfmodel.compile({
      optimizer: "rmsprop",
      loss: 'meanSquaredError',
      metrics: ['accuracy'],
	});
}
async function train(){
	var x=tf.tensor(trainimages,[trainDataSize,28,28,1]);
	var y=tf.tensor(trainlabels,[trainDataSize,10]);
	var iterations=50;
	for(var i=0;i<iterations;i++){
		var result=await tfmodel.fit(x,y,{epochs:1,validationData:null});
		console.log("Epochs= "+i+"Loss= "+result.history.loss[0]);
	}
	console.log("Training Done");
	tfmodel.save("downloads://digitsmodel");
	test();
	// manualtest();
}
function find_max(a){
	var max=a[0];
	var max_index=0;
	for(var i=1;i<a.length;i++){
		if(a[i]>max){
			max=a[i];
			max_index=i;
		}
	}
	return max_index;
}
async function manualtest(){
	var correct=0;
	var total=0;
	for(var i=0;i<testDataSize;i++){
		total++;
		var x=tf.tensor(testimages[i],[1,28,28,1]);
		var resultTensor=tfmodel.predict(x);
		var result=await resultTensor.data();
		var max_index=find_max(result);
		if(max_index==find_max(testlabels[i]))
			correct++;
	}
	console.log("Accuracy= "+((correct/total)*100));
}
async function predict(){
	var index=0;
	img=get();
	img.resize(28,28);
	img.loadPixels();
	var x=[];
	for(var i=0;i<784;i++){
		x.push(img.pixels[index]/255);
		index+=4;
	}
	var tx=tf.tensor(x,[1,28,28,1]);
	var resultTensor=tfmodel.predict(tx);
	resultTensor.print();
	var result=await resultTensor.data();
	console.log("Digit is= "+find_max(result));
	document.getElementById("result").innerHTML="Digit is "+find_max(result);
}
function clearScreen(){
	background(0);
}
async function loadtfModel(){
	tfmodel=await tf.loadLayersModel('digitsmodel.json');
	console.log("Model Loaded");
}
function test(){
	var x=tf.tensor(testimages,[testDataSize,28,28,1]);
	var y=tf.tensor(testlabels,[testDataSize,10]);
	var result=tfmodel.evaluate(x,y);
	var acc = result[1].dataSync()[0] * 100;
	console.log("Accuracy= "+acc);
}
function setup(){
	createData();
	// createtfmodel();
	// train();
	loadtfModel();
	createCanvas(280,280);
	background(0);
	// img=createImage(28,28);
	// img.loadPixels();
	// var index=0;
	// for(var i=0;i<784;i++){
	// 	img.pixels[index]=trainimages[59999][i];
	// 	img.pixels[index+1]=trainimages[59999][i];
	// 	img.pixels[index+2]=trainimages[59999][i];
	// 	img.pixels[index+3]=255;
	// 	index+=4;
	// }
	// img.updatePixels();
	// img.resize(280,280);
	// image(img,0,0);
}
function draw(){
	strokeWeight(8);
	stroke(255);
	if(mouseIsPressed){
		line(pmouseX,pmouseY,mouseX,mouseY);
	}
}