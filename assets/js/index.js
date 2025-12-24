window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

});

// --- Dark Mode & Bilingual Logic ---
document.addEventListener('DOMContentLoaded', () => {
  // 1. Dark Mode Logic (Removed)
  /*
  const themeToggleBtn = document.getElementById('theme-toggle');
  const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
  
  // Get saved theme or default to system
  const currentTheme = localStorage.getItem('theme');
  
  if (currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    if(themeToggleBtn) themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
  } else if (currentTheme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    if(themeToggleBtn) themeToggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
  } else {
    // Follow system
    if (prefersDarkScheme.matches) {
      document.documentElement.setAttribute('data-theme', 'dark');
      if(themeToggleBtn) themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
    }
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      let theme = document.documentElement.getAttribute('data-theme');
      if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
        themeToggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
      } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
        themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
      }
    });
  }
  */

  // 2. Language Logic
  const langToggleBtn = document.getElementById('lang-toggle');
  const savedLang = localStorage.getItem('lang') || 'en'; // Default English
  
  document.body.classList.add(`lang-${savedLang}`);
  updateLangButton(savedLang);

  if (langToggleBtn) {
    langToggleBtn.addEventListener('click', () => {
      if (document.body.classList.contains('lang-en')) {
        setLanguage('zh');
      } else {
        setLanguage('en');
      }
    });
  }

  function setLanguage(lang) {
    document.body.classList.remove('lang-en', 'lang-zh');
    document.body.classList.add(`lang-${lang}`);
    localStorage.setItem('lang', lang);
    updateLangButton(lang);
  }

  function updateLangButton(lang) {
    if (langToggleBtn) {
      langToggleBtn.innerText = lang === 'en' ? 'CN' : 'EN'; // Show target language
    }
  }
});
