$(function(){

    $('#keywordsubmit').on('click', (async function(){
	
        tabs =  await chrome.tabs.query({active: true, lastFocusedWindow: true})
        let url = tabs[0].url;
        
        if (url){
            chrome.runtime.sendMessage(
                {url: url},
                function(response) {
                    result = response.farewell;
                    console.log(result.message);
                    $('#text').val(result.message);
                });
        }
    }));
});