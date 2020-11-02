function createDropzone(url_recognize)
{
  Dropzone.autoDiscover = false;
  Dropzone.options.dzone = {
          acceptedFiles: "*.jpeg, *.jpg"
          }

  $(function() {
    var myDropzone = new Dropzone("#my-dropzone");
    myDropzone.on("queuecomplete", function(file) {
      // Called when all files in the queue finish uploading.

      document.getElementById("img-container").style.visibility = "visible";
      window.location = url_recognize;
    });
  })
}


function createCropper(cropData)
{

window.addEventListener('DOMContentLoaded', function () {

      console.log("DOMContentLoaded!");

      var image = document.querySelector('#image');
      console.log("create new cropper", cropData);
      var cropper = new Cropper(image, {
        aspectRatio: 1,
        data: cropData,
        movable: false,
        zoomable: false,
        rotatable: false,
        scalable: false
      });
      console.log(cropper);

      document.getElementById('recognize').onclick = function () {
       console.log("recognize clicked!");

        document.getElementById('secret').value = JSON.stringify(cropper.getData());
        cropper.destroy();
        document.getElementById("send_crop").submit();
      };
    });
}

function addPredictions(pred_files, pred_names, probs, img_path)
{

if (pred_files != ""){

   var i;
   for (i = 0; i < pred_files.length; i ++)
      {
      console.log(pred_files[i], pred_names[i]);
      var divItem = document.createElement("div", id="div" + pred_names[i]);

      var imgItem = document.createElement("img", id="img" + pred_names[i]);
      imgItem.src = img_path + pred_files[i];
      imgItem.width = 100;
      var textItem = document.createElement("p", "p" + pred_names[i], src=pred_files[i]);
      textItem.innerHTML = pred_names[i] + ": " + probs[i] + "%";
      divItem.appendChild(textItem);
      divItem.appendChild(imgItem);
      divItem.id = "div" + pred_names[i];
      document.getElementById("predictions").appendChild(divItem);

      }

   }

   }


function drawBorder(x, y, cropData, img, img_width, img_height, ctx, orig_width, orig_height)
{


    // Top left corner of crop area, rescaled from original image size to shown in browser
    x0 = cropData.x * img.width / img_width;
    y0 = cropData.y * img.height / img_height;

    // Width and height of crop area, also rescaled
    width0 = cropData.width * img.width / img_width;
    height0 = cropData.height * img.height / img_height;

    if ((x != "") && (y != ""))    {

          var scale_x  = width0 / orig_width;
          var scale_y  = height0 / orig_height;

          // Draw the area of the detected flag border
          ctx.beginPath();
          ctx.moveTo(x0 + x[1] * scale_x, y0 + y[1] * scale_y);
          ctx.lineTo(x0 + x[2] * scale_x, y0 + y[2] * scale_y);
          ctx.lineTo(x0 + x[3] * scale_x, y0 + y[3] * scale_y);
          ctx.lineTo(x0 + x[0] * scale_x, y0 + y[0] * scale_y);
          ctx.lineTo(x0 + x[1] * scale_x, y0 + y[1] * scale_y);
          ctx.stroke();


            }
            else

            {
            console.log("no x or y")
            };


};


function getContext(cnv, img){
    cnv.style.position = "absolute";
    cnv.style.left = img.offsetLeft + "px";
    cnv.style.top = img.offsetTop + "px";
    cnv.width = img.width;
    cnv.height = img.height;
    return cnv.getContext("2d");
}

