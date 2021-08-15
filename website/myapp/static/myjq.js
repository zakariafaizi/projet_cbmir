$(document).ready(function(){




$("#loadingdiv").hide();


for(var i = 1 ; i<=20;i++)
{
    var source = '<img class="col-4 imgs" src=../../media/output/'+i+'.png />';

    $("#div1").append(String(source));



}






});

var upload = false;

$("#btnUpload").on("click", function(e){

    if(upload)
    {
        $("#loadingdiv").show();
        $("#form1").hide();
    }

    upload = false;

});

$("#id_medical_Img").on('propertychange input', function (e) {

  upload = true;

});