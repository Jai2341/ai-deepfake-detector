/* WAIT UNTIL PAGE LOADS */
document.addEventListener("DOMContentLoaded", function () {

    /* 🔤 TYPING TITLE */
    const text = "🛡 AI Deepfake Detector";
    let i = 0;
    const typingElement = document.getElementById("typingTitle");

    function typeWriter(){
        if(i < text.length){
            typingElement.innerHTML += text.charAt(i);
            i++;
            setTimeout(typeWriter, 70);
        }
    }

    if(typingElement){
        typeWriter();
    }

    /* ⏳ LOADING EFFECT */
    const form = document.querySelector("form");
    const loader = document.getElementById("loader");

    if(form && loader){
        form.addEventListener("submit", function(){
            loader.classList.remove("hidden");
        });
    }

});