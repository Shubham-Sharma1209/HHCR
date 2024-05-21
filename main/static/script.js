function UploadonClick(){
    // alert("CLicked");
    document.getElementById('fileuploader').click();
}
function readURL(input) {
             
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#up')
                .attr('src', e.target.result)
                .width(550)
                .height(500);
        };

        reader.readAsDataURL(input.files[0]);
    }
}



$(document).ready(function (e) {


$("#fileuploader").change(function(){
      readURL(this);
  });

$('#uploadButton').on('click',(function(e) {
   e.preventDefault();

   var file1 = document.getElementById("fileuploader").files[0];
   
   var formData = new FormData();
   formData.append('imagePath', file1);
  
   $.ajax({
       type:'POST',
       url: 'detect',
       data:formData,
       contentType: 'multipart/form-data',
       cache:false,
       contentType: false,
       processData: false,
       success:function(data){
           console.log("success");
       },
       error: function(data){
           console.log("error");
          
       }
   });
}));

});