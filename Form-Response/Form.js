function validateCaptcha() {
    var captcha = document.getElementById("captcha").value;
    var captchaAnswer = document.getElementById("captchaAnswer").value;

    if (captcha === captchaAnswer) {
        return true;
    } else {
        alert("Invalid captcha!");
        return false;
    }
}

document.getElementById("submit").addEventListener("click", function () {
    if (validateCaptcha()) {
        // Submit the form
    }
});

function validateForm() {
    var name = document.getElementById("name").value;
    var email = document.getElementById("email").value;
    var message = document.getElementById("message").value;
    var image = document.getElementById("image").value;
    var audio = document.getElementById("audio").value;
    var video = document.getElementById("video").value;
    var captcha = document.getElementById("captcha").value;

    if (name == "" || email == "" || message == "" || image == "" || audio == "" || video == "" || captcha == "") {
        alert("Please fill out all of the fields.");
        return false;
    }

    if (message.length > 1000) {
        alert("The message must be 1000 words or less.");
        return false;
    }

    if (captcha.length != 6) {
        alert("The captcha is incorrect.");
        return false;
    }

    return true;
}
