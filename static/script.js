function previewImage(){

let file=document.getElementById("imageUpload").files[0]

let reader=new FileReader()

reader.onload=function(){

document.getElementById("preview").src=reader.result

}

reader.readAsDataURL(file)

}


function uploadImage(){

let fileInput=document.getElementById("imageUpload")

let file=fileInput.files[0]

let formData=new FormData()

formData.append("file",file)

fetch("/predict",{
method:"POST",
body:formData
})

.then(response=>response.json())

.then(data=>{

document.getElementById("result").innerHTML=
"Prediction : "+data.result

})

}