

function parse(){
    var x = document.getElementById("replytext");
    var req = document.getElementById("emailtext").value;
    // console.log(req)
    // x.innerHTML = req;
    // test
    var request = new XMLHttpRequest()

    // Open a new connection, using the GET request on the URL endpoint
    //request.open('POST', 'http://localhost:5000/api/v1/classification/getcontent/sv', true)
    request.open("POST", "http://10.210.0.184:7010/api/v1/classification/getcontent/sv");
    request.setRequestHeader("Content-Type", "application/json");
    request.onload = function() {
        // Begin accessing JSON data here
	var data = JSON.parse(this.response)

	if (request.status >= 200 && request.status < 400) {
            if (document.getElementById("yes_sign").checked) {
	        x.innerHTML = data['content'] + '\n ' + data['signature']
            }
            else if(document.getElementById("no_sign").checked) {
	        x.innerHTML = data['content'] 
            }
            else {
	        x.innerHTML = data['content'] + '\n ' + data['signature']
            }
	} else {
             x.innerHTML = 'error'
        }
    }
    if (document.getElementById("use_nn").checked) {
        var data = JSON.stringify({"text": [req], "use_nn": "True"}); //, "use_nn": "True"
        request.send(data)
    }
    else if (document.getElementById("use_heur").checked) {
        var data = JSON.stringify({"text": [req]}); 
        request.send(data)
    }
    else{
        var data = JSON.stringify({"text": [req], "use_nn": "True"}); //, "use_nn": "True"
        request.send(data)
    }
}
