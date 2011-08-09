<?xml version='1.0'?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version='1.0'>

<!-- ********************************************************************
     $Id: admon.xsl 6434 2006-11-18 09:00:48Z bobstayton $
     ********************************************************************

     This file is part of the XSL DocBook Stylesheet distribution.
     See ../README or http://nwalsh.com/docbook/xsl/ for copyright
     and other information.

     ******************************************************************** -->

<xsl:template match="*" mode="admon.graphic.width">
  <xsl:param name="node" select="."/>
  <xsl:text>25</xsl:text>
</xsl:template>

<!-- Choose a given graphic when the attribute role='pro' is present in a node
element -->
<xsl:template match="note|important|warning|caution|tip">
  <xsl:choose>
    <xsl:when test="$admon.graphics != 0">
      <xsl:choose>
	<xsl:when test="@role = 'pro'">
          <xsl:call-template name="graphical.admonition">
	      <xsl:with-param name="is_pro" select="'yes'"/>
          </xsl:call-template>
	</xsl:when>
	<xsl:otherwise>
          <xsl:call-template name="graphical.admonition">
	      <xsl:with-param name="is_pro" select="'no'"/>
          </xsl:call-template>
	</xsl:otherwise>
      </xsl:choose>
    </xsl:when>
    <xsl:otherwise>
      <xsl:call-template name="nongraphical.admonition"/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template name="admon.graphic">
  <xsl:param name="is_pro" select="'no'"/>
  <xsl:param name="node" select="."/>
  <xsl:value-of select="$admon.graphics.path"/>
    <xsl:if test="$is_pro = 'yes'">
      <xsl:choose>
        <xsl:when test="local-name($node)='note'">note_pro</xsl:when>
        <xsl:when test="local-name($node)='warning'">warning_pro</xsl:when>
        <xsl:when test="local-name($node)='caution'">caution_pro</xsl:when>
        <xsl:when test="local-name($node)='tip'">tip_pro</xsl:when>
        <xsl:when test="local-name($node)='important'">important_pro</xsl:when>
        <xsl:otherwise>note_pro</xsl:otherwise>
      </xsl:choose>
    </xsl:if>
    <xsl:if test="$is_pro != 'yes'">
      <xsl:choose>
        <xsl:when test="local-name($node)='note'">note</xsl:when>
        <xsl:when test="local-name($node)='warning'">warning</xsl:when>
        <xsl:when test="local-name($node)='caution'">caution</xsl:when>
        <xsl:when test="local-name($node)='tip'">tip</xsl:when>
        <xsl:when test="local-name($node)='important'">important</xsl:when>
        <xsl:otherwise>note</xsl:otherwise>
      </xsl:choose>
    </xsl:if>
  <xsl:value-of select="$admon.graphics.extension"/>
</xsl:template>

<!-- The param is_pro is passed when this template is called by the template
which matches note elements. The default param value is 'no'. This template
calls the admon.graphic template with the current is_pro param value. -->
<xsl:template name="graphical.admonition">
  <xsl:param name="is_pro"/>
  <xsl:variable name="admon.type">
    <xsl:choose>
      <xsl:when test="local-name(.)='note'">Note</xsl:when>
      <xsl:when test="local-name(.)='warning'">Warning</xsl:when>
      <xsl:when test="local-name(.)='caution'">Caution</xsl:when>
      <xsl:when test="local-name(.)='tip'">Tip</xsl:when>
      <xsl:when test="local-name(.)='important'">Important</xsl:when>
      <xsl:otherwise>Note</xsl:otherwise>
    </xsl:choose>
  </xsl:variable>

  <xsl:variable name="alt">
    <xsl:call-template name="gentext">
      <xsl:with-param name="key" select="$admon.type"/>
    </xsl:call-template>
  </xsl:variable>

  <div>
    <xsl:if test="$admon.style != ''">
      <xsl:attribute name="style">
        <xsl:value-of select="$admon.style"/>
      </xsl:attribute>
    </xsl:if>

    <xsl:apply-templates select="." mode="class.attribute"/>

    <table border="0">
      <xsl:attribute name="summary">
        <xsl:value-of select="$admon.type"/>
        <xsl:if test="title|info/title">
          <xsl:text>: </xsl:text>
          <xsl:value-of select="(title|info/title)[1]"/>
        </xsl:if>
      </xsl:attribute>
      <tr>
        <td rowspan="2" align="center" valign="top">
          <xsl:attribute name="width">
            <xsl:apply-templates select="." mode="admon.graphic.width"/>
          </xsl:attribute>
          <img alt="[{$alt}]">
            <xsl:attribute name="src">
              <xsl:call-template name="admon.graphic">
		 <xsl:with-param name="is_pro" select="$is_pro"/>
              </xsl:call-template>
            </xsl:attribute>
          </img>
        </td>
        <th align="left">
          <xsl:call-template name="anchor"/>
          <xsl:if test="$admon.textlabel != 0 or title or info/title">
            <xsl:apply-templates select="." mode="object.title.markup"/>
          </xsl:if>
        </th>
      </tr>
      <tr>
        <td align="left" valign="top">
          <xsl:apply-templates/>
        </td>
      </tr>
    </table>
  </div>
</xsl:template>

<xsl:template name="nongraphical.admonition">
  <div>
    <xsl:if test="$admon.style">
      <xsl:attribute name="style">
        <xsl:value-of select="$admon.style"/>
      </xsl:attribute>
    </xsl:if>

    <xsl:apply-templates select="." mode="class.attribute"/>

    <h3 class="title">
      <xsl:call-template name="anchor"/>
      <xsl:if test="$admon.textlabel != 0 or title or info/title">
        <xsl:apply-templates select="." mode="object.title.markup"/>
      </xsl:if>
    </h3>

    <xsl:apply-templates/>
  </div>
</xsl:template>

<xsl:template match="note/title"></xsl:template>
<xsl:template match="important/title"></xsl:template>
<xsl:template match="warning/title"></xsl:template>
<xsl:template match="caution/title"></xsl:template>
<xsl:template match="tip/title"></xsl:template>

</xsl:stylesheet>
