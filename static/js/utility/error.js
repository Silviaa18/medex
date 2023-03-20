function report_error(message) {
    let div = document.getElementById('error_div');
    div.innerHTML = `
        <div class="alert alert-danger mt-2" role="alert">
            ${message}
            <span id="error_close_button" class="close" aria-label="Close"> &times;</span>
        </div>
    `;
    let close_button = document.getElementById('error_close_button');
    close_button.addEventListener('click', clear_error);
}

function clear_error() {
    let div = document.getElementById('error_div');
    div.innerHTML = null;
}

export {report_error, clear_error};