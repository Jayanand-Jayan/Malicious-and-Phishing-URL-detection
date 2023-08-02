chrome.tabs.executeScript({
    code: "window.location.href.toString();"
}, function(url) {
    let req = new XMLHttpRequest();
    req.open('POST', 'http://localhost:12345/prediction');
    req.setRequestHeader('Content-Type', 'application/json');   
    req.onload = function() {
        let response = JSON.parse(req.responseText);
        if (response.prediction)
            document.getElementById("output").innerHTML = response.prediction;
        else
            document.getElementById("output").innerHTML = response.trace;
    }   
    let sendData = JSON.stringify({
        "review": url
    });
    req.send(sendData);
});