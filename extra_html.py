
def html():
    return"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <img src="https://img001.prntscr.com/file/img001/msO_EYrzSEK8ftxmnaGGgg.png" width="165" height="85"  />
        <title> Lyzer APP</title>
        <style>
            h1{
                font-size: 36px;
                color: cyan;
            }
    </style>
    </head>
    <body>
        
    </body>
    </html>
    """
def fileUpload():
    return"""
    <form>
        <input type="file" id="fileUpload" name="modelname">
    </form>
    
    <script type="text/javascript">

    $(function()
    {
        $('#fileUpload').on('change',function ()
        {
            var filePath = $(this).val();
            console.log(filePath);
        });
    });
    </script>
    """