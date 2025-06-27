document.addEventListener('DOMContentLoaded', function () {
    const prevButton = document.querySelector('.slider-prev');
    const nextButton = document.querySelector('.slider-next');
    const sliderBox = document.querySelector('.slider-box');
    let index = 0;

    // Function to move the slider
    function moveSlider() {
        sliderBox.style.transform = `translateX(-${index * 100}%)`;
    }

    // Event listeners for next and previous buttons
    nextButton.addEventListener('click', function () {
        if (index < sliderBox.children.length - 1) {
            index++;
        } else {
            index = 0; // Loop back to the first slide
        }
        moveSlider();
    });

    prevButton.addEventListener('click', function () {
        if (index > 0) {
            index--;
        } else {
            index = sliderBox.children.length - 1; // Loop to the last slide
        }
        moveSlider();
    });
});
