<?xml version='1.0' encoding="utf-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:fo="http://www.w3.org/1999/XSL/Format"
                version="1.0">

<!-- This customisation layer adds support for the FOP implementation of PDF bookmarks as defined in the latest XSLT draft. It also ensures that the symbol geq is properly rendered in the final PDF document. -->

    <xsl:import href="tldp-print.xsl"/>
    <xsl:import href="bookmarks_fop.xsl"/>
    <xsl:include href="graphical_titlepage.xsl"/>
    <xsl:include href="role_admon.xsl"/>
    <xsl:include href="../common/pytables_common.xsl"/>

    <!-- Render correctly the greater equal symbol -->
    <xsl:template	match="symbol[@role='geq']"><fo:inline font-family="Symbol">&#x2265;</fo:inline></xsl:template>

    <!-- DOUBLE SIDE PRINTING -->
    <xsl:param name="double.sided" select="0"></xsl:param>

    <!-- ADMONITION -->
    <!-- * use graphical admonitions in SVG format -->
    <xsl:param name="admon.graphics" select="1"></xsl:param>
    <xsl:param name="admon.graphics.extension" select="'.svg'"></xsl:param>

    <!-- BODY TEXT -->
    <!-- * fully justify
         * do not indent (two parameters must be setup) -->
    <xsl:param name="alignment">justify</xsl:param>

    <xsl:param name="body.start.indent">
      <xsl:choose>
        <xsl:when test="$fop.extensions != 0">4pt</xsl:when>
         <xsl:when test="$passivetex.extensions != 0">0pt</xsl:when>
         <xsl:otherwise>4pc</xsl:otherwise>
      </xsl:choose>
    </xsl:param>

    <xsl:param name="title.margin.left">
      <xsl:choose>
        <xsl:when test="$fop.extensions != 0">0pc</xsl:when>
        <xsl:when test="$passivetex.extensions != 0">0pt</xsl:when>
        <xsl:otherwise>0pt</xsl:otherwise>
      </xsl:choose>
    </xsl:param>

    <!-- HEADERS -->
    <!-- * give more room for titles than for page numbers -->
    <xsl:param name="header.column.widths" select="'1 2 1'"></xsl:param>

    <!-- CROSS REFERENCES -->
    <!-- * colorize references -->
    <xsl:attribute-set name="xref.properties">
      <xsl:attribute name="color">
        <xsl:choose>
          <xsl:when test="self::ulink">blue</xsl:when>
          <xsl:when test="self::biblioref">#28A000</xsl:when>
          <xsl:otherwise>red</xsl:otherwise>
        </xsl:choose>
      </xsl:attribute>
    </xsl:attribute-set>

    <!-- VERBATIM ENVIRONMENTS -->
    <!-- * make font size smaller than default
         * wrap long lines
         * do not hyphenate (there seems to be a bug) -->
    <xsl:attribute-set name="monospace.verbatim.properties" use-attribute-sets="verbatim.properties monospace.properties">
      <xsl:attribute name="font-size">10pt</xsl:attribute>
      <xsl:attribute name="wrap-option">wrap</xsl:attribute>
      <xsl:attribute name="hyphenation-character">&#x21B2;</xsl:attribute>
    </xsl:attribute-set>

    <!-- GLOSSARIES -->
    <!-- * format glosslists as blocks (this is sometime overwritten with PIs)
         * reduce the separation between terms and defs -->
    <xsl:param name="glosslist.as.blocks" select="1"></xsl:param>
    <xsl:param name="glossterm.separation" select="'0.2in'"></xsl:param>
    <xsl:param name="glossterm.width" select="'2.0in'"></xsl:param>

    <!-- FORMAL OBJECT TITLES (tables and figures) -->
    <!-- * customise title alignment, font size and font weight -->
    <xsl:attribute-set name="formal.title.properties" use-attribute-sets="normal.para.spacing">
      <xsl:attribute name="font-weight">normal</xsl:attribute>
      <xsl:attribute name="font-size">
        <xsl:value-of select="$body.font.master * 1.0"></xsl:value-of>
        <xsl:text>pt</xsl:text>
      </xsl:attribute>
      <xsl:attribute name="hyphenate">false</xsl:attribute>
      <xsl:attribute name="space-after.minimum">0.4em</xsl:attribute>
      <xsl:attribute name="space-after.optimum">0.6em</xsl:attribute>
      <xsl:attribute name="space-after.maximum">0.8em</xsl:attribute>
      <xsl:attribute name="text-align">
        <xsl:choose>
          <xsl:when test="self::table">center</xsl:when>
          <xsl:when test="self::figure">center</xsl:when>
          <xsl:otherwise>left</xsl:otherwise>
    </xsl:choose>
      </xsl:attribute>
    </xsl:attribute-set>

    <!-- SECTIONS -->
    <!-- Customise section title sizes -->
    <xsl:attribute-set name="section.title.level1.properties">
      <xsl:attribute name="font-size">18pt</xsl:attribute>
    </xsl:attribute-set>

    <xsl:attribute-set name="section.title.level2.properties">
      <xsl:attribute name="font-size">15pt</xsl:attribute>
    </xsl:attribute-set>

    <xsl:attribute-set name="section.title.level3.properties">
      <xsl:attribute name="font-size">12pt</xsl:attribute>
    </xsl:attribute-set>

    <xsl:attribute-set name="section.title.level4.properties">
      <xsl:attribute name="font-size">11pt</xsl:attribute>
    </xsl:attribute-set>

    <xsl:attribute-set name="section.title.level5.properties">
      <xsl:attribute name="font-size">9pt</xsl:attribute>
    </xsl:attribute-set>

</xsl:stylesheet>
