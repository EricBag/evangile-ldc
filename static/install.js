/* ============================================================
   Bouton « Installer l'appli » intelligent
   - Android (Chrome) : déclenche l'installation native en 1 clic
   - iPhone (Safari)  : affiche un mini-tutoriel (Apple interdit
                        l'installation automatique)
   - Se cache si l'appli est déjà installée
   ============================================================ */
(function () {
  "use strict";

  // Déjà installée (lancée depuis l'écran d'accueil) → on ne montre rien
  var dejaInstallee =
    window.matchMedia("(display-mode: standalone)").matches ||
    window.navigator.standalone === true;
  if (dejaInstallee) return;

  var zone = document.getElementById("install-zone");
  if (!zone) return;

  var ua = window.navigator.userAgent || "";
  var estIOS = /iphone|ipad|ipod/i.test(ua) ||
    // iPad iPadOS se présente comme un Mac tactile
    (/Macintosh/i.test(ua) && "ontouchend" in document);
  // Navigateur intégré (WhatsApp, Mail, Messenger, Instagram…) sur iOS :
  // l'option « Sur l'écran d'accueil » n'y est pas disponible.
  var estSafari = /^((?!chrome|crios|fxios|edgios|android).)*safari/i.test(ua);
  var estInApp = estIOS && !estSafari && !window.navigator.standalone;

  // Crée le bouton
  var bouton = document.createElement("button");
  bouton.type = "button";
  bouton.className = "bouton-installer";
  bouton.innerHTML =
    '<svg class="bouton-installer-icone" width="17" height="17" viewBox="0 0 24 24" ' +
    'fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></svg>' +
    "<span>Installer l'application</span>";
  bouton.style.display = "none";
  zone.appendChild(bouton);

  // ---------- ANDROID : installation native ----------
  var promptDiffere = null;
  window.addEventListener("beforeinstallprompt", function (e) {
    e.preventDefault();
    promptDiffere = e;
    bouton.style.display = "";
  });

  // ---------- iPhone : tutoriel ----------
  if (estIOS) {
    bouton.style.display = "";
  }

  bouton.addEventListener("click", function () {
    if (promptDiffere) {
      // Android : ouvre la vraie fenêtre d'installation
      promptDiffere.prompt();
      promptDiffere.userChoice.then(function () {
        promptDiffere = null;
        bouton.style.display = "none";
      });
    } else if (estIOS) {
      afficherTutorielIOS();
    }
  });

  // Quand l'appli vient d'être installée → on cache le bouton
  window.addEventListener("appinstalled", function () {
    bouton.style.display = "none";
    fermerTutoriel();
  });

  // ---------- Overlay du tutoriel iPhone ----------
  var overlay = null;

  function afficherTutorielIOS() {
    if (overlay) {
      overlay.style.display = "flex";
      return;
    }
    overlay = document.createElement("div");
    overlay.className = "install-overlay";

    var etapes = estInApp
      ? [
          "Tu as ouvert le lien depuis une messagerie (WhatsApp, Mail…).",
          'Appuie sur <strong>«&nbsp;…&nbsp;»</strong> ou l\'icône de partage, puis sur <strong>«&nbsp;Ouvrir dans Safari&nbsp;»</strong>.',
          "Une fois dans Safari, appuie sur le bouton <strong>Partager</strong> <span class=\"install-share\">⬆️</span> en bas de l'écran.",
          'Choisis <strong>«&nbsp;Sur l\'écran d\'accueil&nbsp;»</strong>, puis <strong>«&nbsp;Ajouter&nbsp;»</strong>.',
        ]
      : [
          "Appuie sur le bouton <strong>Partager</strong> <span class=\"install-share\">⬆️</span> en bas de l'écran.",
          'Fais défiler et choisis <strong>«&nbsp;Sur l\'écran d\'accueil&nbsp;»</strong>.',
          'Appuie sur <strong>«&nbsp;Ajouter&nbsp;»</strong> en haut à droite.',
        ];

    var listeHtml = etapes
      .map(function (t, i) {
        return (
          '<li><span class="install-num">' + (i + 1) + "</span><span>" + t + "</span></li>"
        );
      })
      .join("");

    overlay.innerHTML =
      '<div class="install-modal">' +
      '<h2 class="install-titre">Installer sur iPhone</h2>' +
      '<ol class="install-etapes">' + listeHtml + "</ol>" +
      '<p class="install-note">C\'est une manipulation imposée par Apple : ' +
      "l'installation ne peut pas se faire toute seule sur iPhone.</p>" +
      '<button type="button" class="install-fermer">J\'ai compris</button>' +
      "</div>";

    overlay.addEventListener("click", function (e) {
      if (e.target === overlay || e.target.classList.contains("install-fermer")) {
        fermerTutoriel();
      }
    });

    document.body.appendChild(overlay);
  }

  function fermerTutoriel() {
    if (overlay) overlay.style.display = "none";
  }
})();
