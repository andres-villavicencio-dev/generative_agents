/* =========================================================================
 * RichFX — animation enrichment layer for the Reverie/Generative-Agents
 * frontend (Phaser 3.55.2).
 *
 * Loaded by templates/demo/demo.html and templates/home/home.html AFTER
 * phaser.js and BEFORE the main_script.html include. All textures are
 * generated procedurally (canvas 2D) at init — no new static assets needed.
 *
 * Adds on top of the original tile-stepping renderer:
 *   - Eased per-tile movement (sine ease-in-out) + walk bob/lean + dust
 *   - Idle activity state machine driven by each persona's description
 *     (sleep Zzz, music notes, eat/cook steam, exercise sweat, garden
 *     leaves, wash droplets, work sparkles, talk gestures, ...)
 *   - Typewriter speech bubbles showing real chat utterances
 *   - Pronunciato emoji "pop" emotes on activity change
 *   - Day/night tint from the in-game clock + night lantern glows,
 *     fireflies (night) and drifting leaves (day)
 *   - Artifact pop-in / fade-out
 *
 * Public API (called from the Django templates):
 *   RichFX.init(scene, opts)   — once, at end of create()
 *   RichFX.setClock(dateObj)   — demo mode: JS Date of current in-game time
 *   RichFX.setTimeFromString() — live mode: "February 13, 2023, 09:55:40"
 *   RichFX.onStepData(name, data) — new step payload {x,y,description,chat,pronunciatio}
 *   RichFX.tickMove(name, execute_count) -> 'l'|'r'|'u'|'d'|''  (eased move)
 *   RichFX.tickIdle(name)      — idle FX for absent/stationary personas
 *   RichFX.popIn(obj) / RichFX.popOut(obj) — artifact sprites
 *   RichFX.update()            — once per frame from scene update()
 *   RichFX.debug()             — introspection for QA
 * ========================================================================= */

// ---- error trap (QA: read window.__fxErrors from the console) ----
window.addEventListener('error', function (e) {
  (window.__fxErrors = window.__fxErrors || []).push(
    (e.message || 'error') + ' @' + (e.lineno || '?'));
});
window.addEventListener('unhandledrejection', function (e) {
  (window.__fxErrors = window.__fxErrors || []).push('promise: ' + e.reason);
});

window.RichFX = (function () {
  'use strict';

  var SC = null;            // Phaser scene
  var OPTS = {};           // options from init
  var S = {};              // per-persona state
  var BUBBLES = [];        // active speech bubbles
  var FLOATIES = [];       // floating images (zzz, notes, emotes, ...)
  var EMITTERS = {};       // Phaser particle emitters (manual mode)
  var nightRect = null;    // full-map multiply-tint rectangle
  var hour = 12;           // current in-game hour (float)
  var nightAmount = 0;     // 0..1 darkness
  var lastChatKey = '';    // dedupe speech bubbles across personas
  var ambientNext = { firefly: 0, leaf: 0 };

  // ------------------------------------------------------------------
  // helpers
  // ------------------------------------------------------------------
  function easeInOutSine(t) { return -(Math.cos(Math.PI * t) - 1) / 2; }
  // The discrete tick loop ends one update before t would reach 1.0, so a
  // raw easeInOutSine leaves a ~15% position gap that gets hard-snapped.
  // Rescale so the final tick lands exactly on the target.
  function easeMove(t, cycles) {
    var T = (cycles > 1) ? (1 - 1 / cycles) : 1;
    return easeInOutSine(clamp(t / T, 0, 1));
  }
  function rand(a, b) { return a + Math.random() * (b - a); }
  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }

  function persona(name) { return OPTS.personas[name]; }
  function centerX(name) { return persona(name).body.x + 15; }
  function headY(name) { return persona(name).body.y - 8; }
  function feetY(name) { return persona(name).body.y + 40; }

  // ------------------------------------------------------------------
  // procedural textures (canvas 2D → Phaser textures)
  // ------------------------------------------------------------------
  function addCanvas(key, canvas) {
    if (!SC.textures.exists(key)) SC.textures.addCanvas(key, canvas);
  }

  function makeRadial(key, size, stops) {
    if (SC.textures.exists(key)) return key;
    var c = document.createElement('canvas');
    c.width = size; c.height = size;
    var ctx = c.getContext('2d');
    var g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    for (var i = 0; i < stops.length; i++) g.addColorStop(stops[i][0], stops[i][1]);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    addCanvas(key, c);
    return key;
  }

  function makeSpark(key) {
    if (SC.textures.exists(key)) return key;
    var c = document.createElement('canvas');
    c.width = 24; c.height = 24;
    var ctx = c.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(12, 0); ctx.quadraticCurveTo(13.5, 8.5, 24, 12);
    ctx.quadraticCurveTo(13.5, 15.5, 12, 24);
    ctx.quadraticCurveTo(10.5, 15.5, 0, 12);
    ctx.quadraticCurveTo(10.5, 8.5, 12, 0);
    ctx.closePath(); ctx.fill();
    addCanvas(key, c);
    return key;
  }

  function makeNote(key, glyph) {
    if (SC.textures.exists(key)) return key;
    var c = document.createElement('canvas');
    c.width = 36; c.height = 44;
    var ctx = c.getContext('2d');
    ctx.font = '34px Georgia, serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.lineWidth = 5; ctx.strokeStyle = 'rgba(30,30,60,0.85)';
    ctx.strokeText(glyph, 18, 24);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(glyph, 18, 24);
    addCanvas(key, c);
    return key;
  }

  function makeLeaf(key) {
    if (SC.textures.exists(key)) return key;
    var c = document.createElement('canvas');
    c.width = 22; c.height = 22;
    var ctx = c.getContext('2d');
    ctx.fillStyle = '#7cb342';
    ctx.beginPath();
    ctx.ellipse(11, 11, 9, 4.5, Math.PI / 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#558b2f'; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(4, 16); ctx.lineTo(18, 6); ctx.stroke();
    addCanvas(key, c);
    return key;
  }

  function makeDrop(key) {
    if (SC.textures.exists(key)) return key;
    var c = document.createElement('canvas');
    c.width = 14; c.height = 18;
    var ctx = c.getContext('2d');
    ctx.fillStyle = '#4fc3f7';
    ctx.beginPath();
    ctx.moveTo(7, 0);
    ctx.quadraticCurveTo(13, 10, 12, 13);
    ctx.arc(7, 13, 5.4, 0, Math.PI, false);
    ctx.quadraticCurveTo(1, 10, 7, 0);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.beginPath(); ctx.arc(5, 13, 1.6, 0, Math.PI * 2); ctx.fill();
    addCanvas(key, c);
    return key;
  }

  function makeEmojiTex(emoji) {
    var codes = '';
    for (var i = 0; i < emoji.length; i++) codes += emoji.charCodeAt(i) + '_';
    var key = 'fxe_' + codes;
    if (SC.textures.exists(key)) return key;
    var c = document.createElement('canvas');
    c.width = 64; c.height = 64;
    var ctx = c.getContext('2d');
    ctx.font = '46px "Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(emoji, 32, 36);
    addCanvas(key, c);
    return key;
  }

  function buildTextures() {
    makeRadial('fx_dust', 24, [[0, 'rgba(190,175,150,0.75)'], [1, 'rgba(190,175,150,0)']]);
    makeRadial('fx_steam', 32, [[0, 'rgba(255,255,255,0.85)'], [1, 'rgba(255,255,255,0)']]);
    makeRadial('fx_glow', 128, [[0, 'rgba(255,214,150,0.9)'], [0.4, 'rgba(255,190,110,0.35)'], [1, 'rgba(255,180,90,0)']]);
    makeRadial('fx_firefly', 16, [[0, 'rgba(220,255,140,1)'], [1, 'rgba(190,255,120,0)']]);
    makeSpark('fx_spark');
    makeNote('fx_note1', '\u266A');   // ♪
    makeNote('fx_note2', '\u266B');   // ♫
    makeLeaf('fx_leaf');
    makeDrop('fx_drop');
    makeEmojiTex('\uD83D\uDCA4');     // 💤 (preload the most common one)
  }

  // ------------------------------------------------------------------
  // emitters (all manual: frequency -1, fired via emitParticleAt)
  // ------------------------------------------------------------------
  function makeEmitter(texKey, config, depth) {
    // Phaser 3.55.x API: add.particles(key) -> ParticleEmitterManager
    // (the emitter-as-GameObject API is 3.60+ only)
    var mgr = SC.add.particles(texKey);
    mgr.setDepth(depth);
    return mgr.createEmitter(config);
  }

  function buildEmitters() {
    EMITTERS.dust = makeEmitter('fx_dust', {
      frequency: -1, lifespan: { min: 320, max: 520 },
      speed: { min: 6, max: 26 }, angle: { min: 210, max: 330 },
      scale: { start: 0.8, end: 0.2 }, alpha: { start: 0.6, end: 0 }
    }, 1.5);

    EMITTERS.steam = makeEmitter('fx_steam', {
      frequency: -1, lifespan: { min: 1300, max: 1900 },
      speed: { min: 8, max: 18 }, angle: { min: 250, max: 290 },
      scale: { start: 0.45, end: 1.6 }, alpha: { start: 0.4, end: 0 }
    }, 1.5);

    EMITTERS.spark = makeEmitter('fx_spark', {
      frequency: -1, lifespan: { min: 500, max: 800 },
      speed: { min: 18, max: 65 }, scale: { start: 0.9, end: 0.1 },
      alpha: { start: 1, end: 0 }, rotate: { start: 0, end: 180 },
      blendMode: 'ADD'
    }, 2.6);

    EMITTERS.drop = makeEmitter('fx_drop', {
      frequency: -1, lifespan: { min: 700, max: 1000 },
      speed: { min: 28, max: 58 }, angle: { min: 70, max: 110 },
      alpha: { start: 0.95, end: 0 }, scale: { start: 0.9, end: 0.7 }
    }, 1.5);

    EMITTERS.firefly = makeEmitter('fx_firefly', {
      frequency: -1, lifespan: { min: 3500, max: 6000 },
      speed: { min: 4, max: 13 }, scale: { start: 0.9, end: 0.15 },
      alpha: { start: 0.9, end: 0 }, gravityY: -2, blendMode: 'ADD'
    }, 2.6);

    EMITTERS.leaf = makeEmitter('fx_leaf', {
      frequency: -1, lifespan: 7500,
      speed: { min: 7, max: 15 }, angle: { min: 78, max: 102 },
      gravityY: 7, rotate: { start: 0, end: 340 },
      alpha: { start: 0.85, end: 0.1 }, scale: { start: 1, end: 0.85 }
    }, 1.5);
  }

  // ------------------------------------------------------------------
  // day/night tint
  // ------------------------------------------------------------------
  // [hour, r, g, b, alpha] — multiply-blend tint keyframes
  var SKY = [
    [0.0,   12,  26,  66, 0.44],
    [4.6,   12,  26,  66, 0.44],
    [6.4,  255, 176, 120, 0.16],
    [8.0,  255, 255, 255, 0.00],
    [16.6, 255, 255, 255, 0.00],
    [18.4, 255, 152,  74, 0.18],
    [19.9,  52,  66, 138, 0.30],
    [21.3,  12,  26,  66, 0.44],
    [24.0,  12,  26,  66, 0.44]
  ];

  function skyAt(h) {
    for (var i = 1; i < SKY.length; i++) {
      if (h <= SKY[i][0]) {
        var a = SKY[i - 1], b = SKY[i];
        var t = (h - a[0]) / (b[0] - a[0]);
        function L(x, y) { return Math.round(x + (y - x) * t); }
        return { r: L(a[1], b[1]), g: L(a[2], b[2]), b: L(a[3], b[3]),
                 a: a[4] + (b[4] - a[4]) * t };
      }
    }
    var z = SKY[SKY.length - 1];
    return { r: z[1], g: z[2], b: z[3], a: z[4] };
  }

  // ------------------------------------------------------------------
  // activity classification (from the description string)
  // ------------------------------------------------------------------
  function classify(desc) {
    if (!desc) return 'idle';
    if (desc.indexOf('sleep') >= 0) return 'sleep';
    if (desc.indexOf('convers') >= 0 || desc.indexOf(' chat') >= 0 ||
        desc.indexOf('talking') >= 0 || desc.indexOf('discussing') >= 0) return 'talk';
    if (desc.indexOf('piano') >= 0 || desc.indexOf('music') >= 0 ||
        desc.indexOf('guitar') >= 0 || desc.indexOf('sing') >= 0 ||
        desc.indexOf('band practice') >= 0) return 'music';
    if (desc.indexOf('cook') >= 0 || desc.indexOf('baking') >= 0) return 'cook';
    if (desc.indexOf('eat') >= 0 || desc.indexOf('lunch') >= 0 ||
        desc.indexOf('dinner') >= 0 || desc.indexOf('breakfast') >= 0 ||
        desc.indexOf('snack') >= 0 || desc.indexOf('meal') >= 0) return 'eat';
    if (desc.indexOf('drink') >= 0 || desc.indexOf('coffee') >= 0 ||
        desc.indexOf('tea') >= 0) return 'drink';
    if (desc.indexOf('exercis') >= 0 || desc.indexOf('jogging') >= 0 ||
        desc.indexOf(' jog') >= 0 || desc.indexOf('running') >= 0 ||
        desc.indexOf('yoga') >= 0 || desc.indexOf('stretch') >= 0 ||
        desc.indexOf('work out') >= 0 || desc.indexOf('workout') >= 0) return 'exercise';
    if (desc.indexOf('garden') >= 0 || desc.indexOf('plant') >= 0 ||
        desc.indexOf('watering') >= 0) return 'garden';
    if (desc.indexOf('shower') >= 0 || desc.indexOf(' bath') >= 0 ||
        desc.indexOf('brush') >= 0 || desc.indexOf('wash') >= 0) return 'wash';
    if (desc.indexOf('paint') >= 0 || desc.indexOf('draw') >= 0 ||
        desc.indexOf('sketch') >= 0) return 'paint';
    if (desc.indexOf('clean') >= 0 || desc.indexOf('tidy') >= 0 ||
        desc.indexOf('organiz') >= 0) return 'clean';
    if (desc.indexOf('computer') >= 0 || desc.indexOf('laptop') >= 0 ||
        desc.indexOf('coding') >= 0 || desc.indexOf('serv') >= 0 ||
        desc.indexOf('work') >= 0 || desc.indexOf('writ') >= 0 ||
        desc.indexOf('study') >= 0 || desc.indexOf('read') >= 0 ||
        desc.indexOf('research') >= 0 || desc.indexOf('class') >= 0 ||
        desc.indexOf('homework') >= 0 || desc.indexOf('paper') >= 0 ||
        desc.indexOf('grade') >= 0 || desc.indexOf('lesson') >= 0 ||
        desc.indexOf('email') >= 0) return 'work';
    if (desc.indexOf('tv') >= 0 || desc.indexOf('movie') >= 0 ||
        desc.indexOf('watch') >= 0) return 'watch';
    if (desc.indexOf('shop') >= 0 || desc.indexOf('buy') >= 0 ||
        desc.indexOf('purchas') >= 0) return 'shop';
    return 'idle';
  }

  var STATE_EMO = {
    talk: '\uD83D\uDCAC',      // 💬
    cook: '\uD83C\uDF73',      // 🍳
    eat: '\uD83C\uDF7D\uFE0F', // 🍽️
    drink: '\u2615',            // ☕
    exercise: '\uD83D\uDCA6',  // 💦
    garden: '\uD83C\uDF3F',    // 🌿
    wash: '\uD83D\uDCA7',      // 💧
    paint: '\uD83C\uDFA8',     // 🎨
    clean: '\u2728',            // ✨
    work: '\uD83D\uDCDD',      // 📝
    watch: '\uD83D\uDCFA',     // 📺
    shop: '\uD83D\uDECD\uFE0F' // 🛍️
  };

  function setState(name, s) {
    var st = S[name];
    if (st.state === s) return;
    st.state = s;
    st.nextEmit = 0;
    st.noteFlip = false;
    var emo = STATE_EMO[s];
    if (emo && st.seenData) popEmote(name, emo);
  }

  // ------------------------------------------------------------------
  // floating one-shot images (zzz, notes, emotes)
  // ------------------------------------------------------------------
  function floaty(name, texKey, o) {
    o = o || {};
    var x = centerX(name) + (o.jx ? rand(-o.jx, o.jx) : 0);
    var y = headY(name) + (o.dy0 || -6);
    var img = SC.add.image(x, y, texKey).setDepth(o.depth || 3.6);
    var dur = o.dur || 1800;
    img.setScale(o.scale || 1);
    img.alpha = o.alpha == null ? 1 : o.alpha;
    SC.tweens.add({ targets: img, y: y - (o.rise || 24), duration: dur, ease: 'Sine.easeOut' });
    if (o.sway) {
      SC.tweens.add({ targets: img, x: x + o.sway, duration: dur / 2,
                      ease: 'Sine.easeInOut', yoyo: true, repeat: 1 });
    }
    SC.tweens.add({ targets: img, alpha: 0, delay: dur * 0.55, duration: dur * 0.45,
                    onComplete: function () { img.destroy(); } });
    FLOATIES.push(img);
    if (FLOATIES.length > 80) FLOATIES.splice(0, FLOATIES.length - 80);
  }

  function popEmote(name, emoji) {
    var key = makeEmojiTex(emoji);
    var x = centerX(name) + rand(-8, 8);
    var y = headY(name) - 14;
    var img = SC.add.image(x, y, key).setDepth(3.5).setScale(0);
    SC.tweens.add({ targets: img, scale: 1, duration: 240, ease: 'Back.easeOut' });
    SC.tweens.add({ targets: img, y: y - 18, duration: 1900, ease: 'Sine.easeOut' });
    SC.tweens.add({ targets: img, alpha: 0, delay: 1450, duration: 450,
                    onComplete: function () { img.destroy(); } });
    FLOATIES.push(img);
  }

  // ------------------------------------------------------------------
  // speech bubbles (typewriter, follows the speaker)
  // ------------------------------------------------------------------
  function showSpeech(speakerName, text) {
    // cap concurrent bubbles
    while (BUBBLES.length >= 3) killBubble(BUBBLES[0]);

    var txt = SC.add.text(0, 0, '', {
      fontFamily: '"Segoe UI", Verdana, sans-serif',
      fontSize: '13px', color: '#1a1a2e',
      wordWrap: { width: 210 }, align: 'left',
      padding: { x: 6, y: 4 }
    }).setDepth(3.45);
    txt.setText(text);                 // measure full size
    var w = clamp(txt.width, 64, 236);
    var h = txt.height + 2;
    txt.setText('');                   // start typewriter from empty

    var gfx = SC.add.graphics().setDepth(3.4);
    var b = {
      persona: speakerName, text: text, txt: txt, gfx: gfx,
      w: w, h: h, born: SC.time.now, shown: 0,
      life: 2600 + text.length * 42, dead: false, below: false
    };
    BUBBLES.push(b);
    if (OPTS.onBubbleVisibility) OPTS.onBubbleVisibility(speakerName, true);
  }

  function killBubble(b) {
    if (b.dead) return;
    b.dead = true;
    b.txt.destroy();
    b.gfx.destroy();
    var i = BUBBLES.indexOf(b);
    if (i >= 0) BUBBLES.splice(i, 1);
    if (OPTS.onBubbleVisibility) OPTS.onBubbleVisibility(b.persona, false);
  }

  function drawBubble(b, cx, cy) {
    var g = b.gfx;
    g.clear();
    var below = b.below;
    var bx = clamp(cx - b.w / 2, 8, OPTS.mapWidth - b.w - 8);
    var by = below ? cy + 26 : cy - b.h - 26;
    if (by < 6 && !below) { b.below = true; by = cy + 26; }
    var tailX = clamp(cx, bx + 16, bx + b.w - 16);
    g.fillStyle(0xffffff, 0.94);
    g.fillRoundedRect(bx, by, b.w, b.h, 12);
    g.lineStyle(2, 0x5a6b7c, 0.75);
    g.strokeRoundedRect(bx, by, b.w, b.h, 12);
    g.fillStyle(0xffffff, 0.94);
    if (below) {
      g.fillTriangle(tailX - 7, by - 1, tailX + 7, by - 1, tailX, by + 10);
    } else {
      g.fillTriangle(tailX - 7, by + b.h + 1, tailX + 7, by + b.h + 1, tailX, by + b.h + 12);
    }
    b.txt.setPosition(bx + 6, by + 2);
  }

  function tickBubbles(now) {
    for (var i = BUBBLES.length - 1; i >= 0; i--) {
      var b = BUBBLES[i];
      var p = persona(b.persona);
      if (!p) { killBubble(b); continue; }
      var age = now - b.born;
      // typewriter
      var want = Math.min(b.text.length, Math.floor(age / 26));
      if (want !== b.shown) { b.shown = want; b.txt.setText(b.text.substring(0, want)); }
      // follow speaker
      drawBubble(b, p.body.x + 15, p.body.y - 8);
      // expire
      if (age > b.life) {
        var fade = 1 - (age - b.life) / 280;
        if (fade <= 0) { killBubble(b); continue; }
        b.txt.setAlpha(fade);
        b.gfx.setAlpha(Math.max(0, fade));
      } else { b.txt.setAlpha(1); b.gfx.setAlpha(1); }
    }
  }

  // ------------------------------------------------------------------
  // idle state machine (called every frame for stationary personas)
  // ------------------------------------------------------------------
  function tickState(name) {
    var st = S[name];
    if (!st) return;
    var sp = persona(name);
    var now = SC.time.now;

    switch (st.state) {
      case 'sleep':
        sp.rotation = 0;
        var br = Math.sin(now / 900);
        sp.scaleY = st.baseSY * (0.965 + br * 0.028);
        sp.scaleX = st.baseSX * (1.012 - br * 0.012);
        sp.setAlpha(0.85);
        if (now >= st.nextEmit) {
          st.nextEmit = now + 1500;
          floaty(name, 'fxe_55357_56404_', { rise: 22, dur: 2300, sway: 9, jx: 10 });
        }
        break;

      case 'music':
        sp.rotation = Math.sin(now / 280) * 0.055;
        if (now >= st.nextEmit) {
          st.nextEmit = now + 480;
          st.noteFlip = !st.noteFlip;
          floaty(name, st.noteFlip ? 'fx_note1' : 'fx_note2',
                 { rise: 30, dur: 1900, sway: 15, jx: 12 });
        }
        break;

      case 'cook':
        sp.rotation = 0;
        breathe(sp, st, now, 900);
        if (now >= st.nextEmit) {
          st.nextEmit = now + 380;
          EMITTERS.steam.emitParticleAt(centerX(name) + rand(-6, 6), headY(name) + 16, 1);
          if (!st.flip || Math.random() < 0.25)
            EMITTERS.spark.emitParticleAt(centerX(name), headY(name) + 12, 1);
        }
        break;

      case 'eat':
      case 'drink':
        sp.rotation = 0;
        breathe(sp, st, now, 1000);
        if (now >= st.nextEmit) {
          st.nextEmit = now + (st.state === 'eat' ? 900 : 1300);
          EMITTERS.steam.emitParticleAt(centerX(name) + rand(-4, 4), headY(name) + 14, 1);
          if (st.state === 'eat' && Math.random() < 0.28)
            EMITTERS.spark.emitParticleAt(centerX(name), headY(name) + 10, 1);
        }
        break;

      case 'exercise':
        sp.rotation = 0;
        var pump = Math.sin(now / 160);
        sp.scaleY = st.baseSY * (1 + pump * 0.05);
        sp.scaleX = st.baseSX * (1 - pump * 0.04);
        sp.setAlpha(1);
        if (now >= st.nextEmit) {
          st.nextEmit = now + 900;
          EMITTERS.drop.emitParticleAt(centerX(name) + rand(-8, 8), headY(name) + 6, 1);
        }
        break;

      case 'garden':
        sp.rotation = 0;
        breathe(sp, st, now, 1000);
        if (now >= st.nextEmit) {
          st.nextEmit = now + 2100;
          floaty(name, 'fx_leaf', { rise: -26, dur: 2200, sway: 18, jx: 14 });
          if (Math.random() < 0.4)
            EMITTERS.spark.emitParticleAt(centerX(name) + rand(-14, 14), feetY(name) - 14, 2);
        }
        break;

      case 'wash':
        sp.rotation = 0;
        breathe(sp, st, now, 800);
        if (now >= st.nextEmit) {
          st.nextEmit = now + 620;
          EMITTERS.drop.emitParticleAt(centerX(name) + rand(-10, 10), headY(name) + 10, 1);
        }
        break;

      case 'talk':
        sp.rotation = 0;
        var nod = Math.sin(now / 420);
        sp.scaleY = st.baseSY * (1 + nod * 0.022);
        sp.scaleX = st.baseSX * (1 - nod * 0.018);
        sp.setAlpha(1);
        break;

      case 'work':
      case 'paint':
      case 'clean':
      case 'shop':
      case 'watch':
        sp.rotation = 0;
        breathe(sp, st, now, 1100);
        if (now >= st.nextEmit) {
          st.nextEmit = now + (st.state === 'paint' ? 2600 : 3800);
          EMITTERS.spark.emitParticleAt(centerX(name) + rand(-10, 12), headY(name) + rand(2, 16), 1);
        }
        break;

      default:  // idle
        sp.rotation = 0;
        breathe(sp, st, now, 1300);
        if (now >= st.nextEmit) {
          st.nextEmit = now + 7000;
          if (Math.random() < 0.5)
            EMITTERS.spark.emitParticleAt(centerX(name) + rand(-12, 12), headY(name) + rand(0, 20), 1);
        }
    }
  }

  function breathe(sp, st, now, period) {
    var br = Math.sin(now / period);
    sp.scaleY = st.baseSY * (1 + br * 0.02);
    sp.scaleX = st.baseSX * (1 - br * 0.015);
    sp.setAlpha(1);
  }

  // ------------------------------------------------------------------
  // eased tile movement (replaces the linear body-stepping chain)
  // ------------------------------------------------------------------
  function tickMove(name, execute_count) {
    var st = S[name];
    if (!st) return '';
    var sp = persona(name);
    if (!sp) return '';

    if (!st.hasTarget || execute_count <= 0) {
      if (execute_count <= 0 && st.hasTarget) { sp.body.x = st.tx; sp.body.y = st.ty; }
      tickState(name);
      return '';
    }

    // Phase-wrap guard: when execute_count hits 0, the first persona in the
    // template's loop snaps ALL bodies and resets execute_count to
    // cycles+1 mid-loop; personas later in the loop then reach tickMove
    // with execute_count == cycles+1. Interpolating with that counter
    // clamps t to 0 and yanks the sprite back to the step's START tile —
    // the "walks then bounces back" bug. Hold at the landed target instead.
    if (execute_count > st.cycles) {
      sp.body.x = st.tx; sp.body.y = st.ty;
      st.hasTarget = false;
      tickState(name);
      return '';
    }

    var t = (st.cycles - execute_count) / st.cycles;
    t = clamp(t, 0, 1);
    var moved = (st.tx !== st.sx) || (st.ty !== st.sy);
    var dir = '';
    if (st.tx > st.sx) dir = 'r';
    else if (st.tx < st.sx) dir = 'l';
    else if (st.ty > st.sy) dir = 'd';
    else if (st.ty < st.sy) dir = 'u';

    if (moved) {
      var k = easeMove(t, st.cycles);
      sp.body.x = st.sx + (st.tx - st.sx) * k;
      sp.body.y = st.sy + (st.ty - st.sy) * k;
      // walk juice: hop bob + lean into direction
      var hop = Math.sin(t * Math.PI);
      sp.scaleY = st.baseSY * (1 + hop * 0.05);
      sp.scaleX = st.baseSX * (1 - hop * 0.05);
      sp.rotation = (dir === 'l' ? -1 : (dir === 'r' ? 1 : 0)) * hop * 0.05;
      sp.setAlpha(1);
      st.walkBob = hop;
      // arrival dust puff
      if (t > 0.9 && !st.dusted) {
        st.dusted = true;
        EMITTERS.dust.emitParticleAt(centerX(name), feetY(name) - 2, 3);
      }
    } else {
      sp.scaleX = st.baseSX; sp.scaleY = st.baseSY;
      sp.rotation = 0; st.walkBob = 0;
      tickState(name);
      return '';
    }
    return dir;
  }

  // ------------------------------------------------------------------
  // per-frame global update (night tint, glows, shadows, ambient, bubbles)
  // ------------------------------------------------------------------
  function update() {
    if (!SC) return;
    var now = SC.time.now;

    // day/night tint
    var sky = skyAt(hour);
    nightAmount = sky.a / 0.44;
    nightRect.setFillStyle(Phaser.Display.Color.GetColor(sky.r, sky.g, sky.b), sky.a);

    // per-persona glow + shadow follow
    for (var name in S) {
      var st = S[name];
      var sp = persona(name);
      if (!sp) continue;
      var cx = sp.body.x + 15;
      st.shadow.setPosition(cx, sp.body.y + 42);
      var bobScale = 1 + (st.walkBob || 0) * 0.25;
      st.shadow.setScale(bobScale, 1 - (st.walkBob || 0) * 0.2);
      st.glow.setPosition(cx, sp.body.y + 20);
      st.glow.setAlpha(nightAmount * (st.state === 'sleep' ? 0.10 : 0.30));
    }

    // ambient particles
    var view = SC.cameras.main.worldView;
    if (nightAmount > 0.6) {
      if (now >= ambientNext.firefly) {
        ambientNext.firefly = now + 520;
        EMITTERS.firefly.emitParticleAt(
          view.x + rand(0, view.width), view.y + rand(0, view.height), 2);
      }
    } else if (nightAmount < 0.4) {
      if (now >= ambientNext.leaf) {
        ambientNext.leaf = now + 1050;
        EMITTERS.leaf.emitParticleAt(view.x + rand(0, view.width), view.y - 16, 1);
      }
    }

    tickBubbles(now);
  }

  // ------------------------------------------------------------------
  // public API
  // ------------------------------------------------------------------
  function init(scene, opts) {
    SC = scene;
    OPTS = opts;
    OPTS.mapWidth = OPTS.mapWidth || 4480;
    OPTS.mapHeight = OPTS.mapHeight || 3200;
    S = {};
    BUBBLES = [];
    FLOATIES = [];

    buildTextures();
    buildEmitters();

    // full-map multiply tint rectangle (above foreground layers, below UI)
    nightRect = SC.add.rectangle(
      OPTS.mapWidth / 2, OPTS.mapHeight / 2, OPTS.mapWidth, OPTS.mapHeight,
      0xffffff, 0).setDepth(2.5).setBlendMode(Phaser.BlendModes.MULTIPLY);

    for (var name in OPTS.personas) {
      var sp = OPTS.personas[name];
      S[name] = {
        sx: sp.body.x, sy: sp.body.y,
        tx: sp.body.x, ty: sp.body.y,
        hasTarget: false, cycles: OPTS.cycles || 4,
        state: 'idle', nextEmit: 0, noteFlip: false,
        baseSX: sp.scaleX, baseSY: sp.scaleY,
        dusted: false, walkBob: 0, seenData: false,
        lastPron: null,
        shadow: SC.add.ellipse(sp.body.x + 15, sp.body.y + 42, 32, 11,
                                0x000000, 0.22).setDepth(-0.5),
        glow: SC.add.image(sp.body.x + 15, sp.body.y + 20, 'fx_glow')
                       .setBlendMode(Phaser.BlendModes.ADD)
                       .setDepth(2.55).setScale(0.6).setAlpha(0)
      };
    }

    // hide the (empty) static bubbles until someone actually speaks
    if (OPTS.onBubbleVisibility) {
      for (var n2 in OPTS.personas) OPTS.onBubbleVisibility(n2, false);
    }
  }

  function setClock(dateObj) {
    if (dateObj && dateObj.getHours) {
      hour = dateObj.getHours() + dateObj.getMinutes() / 60;
    }
  }

  function setTimeFromString(str) {
    if (!str) return;
    var m = /(\d{1,2}):(\d{2})(?::(\d{2}))?/.exec(str);
    if (m) hour = parseInt(m[1], 10) + parseInt(m[2], 10) / 60;
  }

  function onStepData(name, data) {
    var st = S[name];
    if (!st || !data) return;
    var sp = persona(name);

    st.sx = sp.body.x; st.sy = sp.body.y;
    st.tx = data.x * OPTS.tileWidth;
    st.ty = data.y * OPTS.tileWidth;
    st.hasTarget = true;
    st.dusted = false;

    // activity state
    var desc = (data.description || '').split(' @ ')[0].toLowerCase();
    setState(name, classify(desc));

    // pronunciato emote pop (skip the very first reading to avoid a
    // screenful of pops on page load)
    if (data.pronunciatio && st.seenData && data.pronunciatio !== st.lastPron) {
      popEmote(name, data.pronunciatio);
    }
    if (data.pronunciatio) st.lastPron = data.pronunciatio;
    st.seenData = true;

    // chat bubbles — the transcript is shared between participants, so
    // only show the newest utterance once, over its speaker
    if (data.chat && data.chat.length) {
      var lastLine = data.chat[data.chat.length - 1];
      if (lastLine && lastLine[0] && lastLine[1] != null) {
        var key = lastLine[0] + '|' + lastLine[1];
        if (key !== lastChatKey) {
          lastChatKey = key;
          var speaker = String(lastLine[0]).replace(/ /g, '_');
          if (OPTS.personas[speaker]) showSpeech(speaker, String(lastLine[1]));
        }
      }
    }
  }

  function popIn(obj) {
    if (!obj) return;
    var sx = obj.scaleX || 1, sy = obj.scaleY || 1;
    obj.setScale(0);
    SC.tweens.add({ targets: obj, scaleX: sx, scaleY: sy,
                    duration: 320, ease: 'Back.easeOut' });
    EMITTERS.spark.emitParticleAt(obj.x, obj.y, 8);
  }

  function popOut(obj) {
    if (!obj) return;
    var sx = obj.scaleX, sy = obj.scaleY;
    SC.tweens.add({ targets: obj, alpha: 0, scaleX: sx * 0.5, scaleY: sy * 0.5,
                    duration: 240, ease: 'Sine.easeIn',
                    onComplete: function () { obj.destroy(); } });
  }

  function debug() {
    var states = {};
    for (var n in S) states[n] = S[n].state;
    var live = 0;
    for (var i = 0; i < FLOATIES.length; i++) if (FLOATIES[i].active) live++;
    return {
      version: '1.0',
      hour: Math.round(hour * 100) / 100,
      night: Math.round(nightAmount * 100) / 100,
      bubbles: BUBBLES.map(function (b) {
        return { persona: b.persona, chars: b.shown, len: b.text.length };
      }),
      floaties: live,
      states: states,
      errors: window.__fxErrors || []
    };
  }

  return {
    init: init,
    setClock: setClock,
    setTimeFromString: setTimeFromString,
    onStepData: onStepData,
    tickMove: tickMove,
    tickIdle: tickState,
    popIn: popIn,
    popOut: popOut,
    update: update,
    debug: debug
  };
})();