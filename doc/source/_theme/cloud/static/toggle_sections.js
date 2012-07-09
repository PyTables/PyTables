/*
 * toggle_sections.js
 * ~~~~~~~~~~~~~~
 *
 * Sphinx JavaScript helper for collapsible sections.
 * looks for sections with css class "html-toggle",
 * with optional additional classes "expanded" or "collapsed"
 * (defaults to "collapsed")
 *
 * :copyright: Copyright 2011 by Assurance Technologies
 * :license: BSD
 *
 * NOTE: while this provides full javascript instrumentation,
 * css styling should be applied to .html-toggle > .html-toggle-button
 */

$(document).ready(function (){
  function init(){
    var jobj = $(this);
    var parent = jobj.parent()

    /* add class for styling purposes */
    jobj.addClass("html-toggle-button");

    /* initialize state */
    _setState(jobj, parent.hasClass("expanded") || _containsHash(parent));

    /* bind toggle callback */
    jobj.click(function (){
      _setState(jobj, !parent.hasClass("expanded"));
    });

    /* check for hash changes - older browsers may not have this evt */
    $(window).bind("hashchange", function () {
      var hash = document.location.hash;
      if(!hash)
        return;
      if(_containsHash(parent))
        _setState(jobj, true);
      var pos = $(hash).position();
      window.scrollTo(pos.left, pos.top);
    });
  }

  function _containsHash(parent){
    var hash = document.location.hash;
    if(!hash)
      return false;
    return parent[0].id == hash.substr(1) || parent.find(hash).length>0;
  }

  function _setState(jobj, expanded){
    var parent = jobj.parent();
    if(expanded){
      parent.addClass("expanded").removeClass("collapsed");
      parent.children().show();
    }else{
      parent.addClass("collapsed").removeClass("expanded");
      parent.children().hide();
      parent.children("span:first-child:empty").show(); /* for :ref: span tag */
      jobj.show();
    }
  }

  $(".html-toggle.section > h2, .html-toggle.section > h3, .html-toggle.section > h4, .html-toggle.section > h5, .html-toggle.section > h6").each(init);

});
