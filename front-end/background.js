var serverhost = 'http://127.0.0.1:8000';

	chrome.runtime.onMessage.addListener(
		function(request, sender, sendResponse) {
		    
			var url = serverhost + '/get_summary/'
		
			fetch(url, {
				method: "POST",
     
    			// Adding body or contents to send
    			body: JSON.stringify({
        			url: request.url
    			}),
    			// Adding headers to the request
    			headers: {
        			"Content-type": "application/json; charset=UTF-8"
    			}
			})
			.then(response => response.json())
			.then(response => sendResponse({farewell: response}))
			.catch(error => console.log(error))
				
			return true;  // Will respond asynchronously.
});

var contextMenuItem = {
    "id": "summaryText",
    "title": "SummaryText",
    "contexts": ["all"]
}

chrome.contextMenus.create(contextMenuItem)
