

function prediction(){
    var city = document.getElementById("city").value
    var g = document.getElementById("sex").value
    var actual_pefr = document.getElementById("pefr").value
    eel.predictor(city,g,actual_pefr)(call_Back)
}

function call_Back(output){
    document.getElementById("op").value =  output
}