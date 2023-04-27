import {switch_nav_item} from "../utility/nav.mjs";

async function init() {
    switch_nav_item('risk_score');
}

document.addEventListener("DOMContentLoaded", init);